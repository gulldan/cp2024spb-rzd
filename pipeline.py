from tqdm import tqdm
import polars as pl
from torch import Tensor
import torch
from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import joblib
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.cluster import HDBSCAN
import warnings
import json

warnings.filterwarnings("ignore")


class ClusterPipe:
    def __init__(self, mtr_path: str, labels_path: str, model_name: str = 'intfloat/multilingual-e5-small'):
        """
        Initialize the Pipeline with paths to the MTR and labels datasets, and the model name.

        Args:
            mtr_path (str): Path to the MTR dataset.
            labels_path (str): Path to the labels dataset.
            model_name (str): Name of the pretrained model to use.
        """
        self.mtr_path = mtr_path
        self.labels_path = labels_path
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1, 2, 3])
        self.model.to('cuda')

    def load_data(self) -> None:
        """
        Load the MTR and labels datasets from the specified paths.
        """
        self.mtr = pl.read_parquet(self.mtr_path)
        self.labels = pl.read_parquet(self.labels_path)
        self.data = self.mtr[["код СКМТР", "Наименование"]].join(self.labels, on="код СКМТР")

    def prepare_texts(self) -> None:
        """
        Prepare the texts for embedding by adding a prefix.
        """
        self.texts = ["passage: " + text for text in self.data["Наименование"].to_numpy()]

    def get_embeddings(self, batch_size: int = 128) -> None:
        """
        Generate embeddings for the given texts in batches.

        Args:
            batch_size (int): Number of texts to process in each batch.
        """
        all_embeddings: list[Tensor] = []
        for i in tqdm(range(0, len(self.texts), batch_size)):
            batch_texts = self.texts[i:i + batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True).to('cuda')
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(embeddings.cpu())
        self.embeddings = torch.cat(all_embeddings, dim=0)

    def save_embeddings(self, output_path: str) -> None:
        """
        Save the embeddings to a Parquet file.

        Args:
            output_path (str): Path to save the embeddings.
        """
        df = pl.DataFrame(self.embeddings.numpy())
        df.write_parquet(output_path)

    def load_embeddings(self, input_path: str) -> None:
        """
        Load the embeddings from a Parquet file.

        Args:
            input_path (str): Path to load the embeddings from.
        """
        self.embeddings = pl.read_parquet(input_path).to_numpy()

    def prepare_for_clustering(self) -> None:
        """
        Prepare the data for clustering by adding the "ОКПД2" and "код СКМТР" columns.
        """
        self.df_for_clustering = pl.DataFrame(self.embeddings).with_columns(
            self.mtr["ОКПД2"].alias("ОКПД2"),
            self.mtr["код СКМТР"].alias("код СКМТР")
        )
        self.df_for_clustering_null = self.df_for_clustering.filter(pl.col("ОКПД2").is_null())
        self.df_for_clustering_not_null = self.df_for_clustering.filter(~(pl.col("ОКПД2").is_null()))

    def encode_labels(self, encoder_path: str) -> None:
        """
        Encode the "ОКПД2" column using a label encoder.

        Args:
            encoder_path (str): Path to the label encoder.
        """
        le = joblib.load(encoder_path)
        self.df_encoded = self.df_for_clustering_not_null.with_columns(
            pl.Series(le.transform(self.df_for_clustering_not_null["ОКПД2"]), dtype=float).alias("ОКПД2")
        )
        self.all_data_encoded = pl.concat(
            [
                self.df_encoded, self.df_for_clustering_null.with_columns(
                pl.Series(np.array([None] * len(self.df_for_clustering_null), dtype=np.float64)).alias("ОКПД2")
            )
            ],
            how="vertical"
        )

    def impute_missing_values(self) -> None:
        """
        Impute missing values using KNNImputer.
        """
        data = self.all_data_encoded.drop(["код СКМТР"]).to_numpy()
        imputer = KNNImputer(n_neighbors=1)
        self.data_imputed = imputer.fit_transform(data)
        self.df_imputed = pl.DataFrame(self.data_imputed)
        self.df_imputed = pl.concat([self.df_imputed, self.all_data_encoded[["код СКМТР"]]], how="horizontal")

    def cluster_data(self, output_path: str, encoder_path: str) -> None:
        """
        Cluster the data for each unique "ОКПД2" value and save the clustered data to a Parquet file.

        Args:
            output_path (str): Path to save the clustered data.
            encoder_path (str): Path to the label encoder.
        """
        unique_okpd2 = self.df_imputed["column_384"].unique().to_numpy()
        full_data = None
        for c, okpd2 in enumerate(tqdm(unique_okpd2)):
            group_df = self.df_imputed.filter(pl.col("column_384") == okpd2)
            features = group_df.drop(["column_384", "код СКМТР"])

            if len(features) < 2:
                group_df = group_df[["column_384", "код СКМТР"]].with_columns(
                    pl.Series([-1] * len(features)).alias("cluster")
                )
            else:
                if len(features) > 1000:
                    print(len(features))
                hdbscan = HDBSCAN(min_cluster_size=2, n_jobs=-1, allow_single_cluster=True)
                group_df = group_df[["column_384", "код СКМТР"]].with_columns(
                    pl.Series(hdbscan.fit_predict(features)).alias("cluster")
                )

            if c == 0:
                full_data = group_df
            else:
                full_data = pl.concat([full_data, group_df], how="vertical")

        full_data.write_parquet(output_path)

        le = joblib.load(encoder_path)
        self.full_data_clustered = full_data.with_columns(
            pl.Series(
                le.inverse_transform(
                    list(map(lambda x: int(x), full_data["column_384"].to_list()))
                )
            ).alias("ОКПД2")
        ).drop(["column_384"])

    def save_clustered_data(self, output_path: str) -> None:
        """
        Save the clustered data to a Parquet file.

        Args:
            output_path (str): Path to save the clustered data.
        """
        self.full_data_clustered.join(self.mtr.drop(["ОКПД2"]), on="код СКМТР").write_parquet(output_path)

    def load_clustered_data(self, input_path: str) -> None:
        """
        Load the clustered data from a Parquet file.

        Args:
            input_path (str): Path to load the clustered data from.
        """
        self.full_data_clustered = pl.read_parquet(input_path).fill_null("")

    def sample_cluster(self, cluster_num: int, okpd_identifier: str) -> pl.DataFrame:
        """
        Filter the clustered data to return a DataFrame containing only the rows that belong to a specific cluster and OKPD2 identifier.

        Args:
            cluster_num (int): The cluster number to filter by.
            okpd_identifier (str): The OKPD2 identifier to filter by.

        Returns:
            pl.DataFrame: A DataFrame containing the rows that match the specified cluster number and OKPD2 identifier.
        """
        return self.full_data_clustered.filter(
            (pl.col("cluster") == cluster_num) & (pl.col("ОКПД2") == okpd_identifier)
        )


class LLM:
    def __init__(self, model_name: str):
        """
        Initialize the LLMParamsFields class with the specified model name.

        Args:
            model_name (str): The name of the LLM model to load.
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.float16)
        self.model.eval()

    def prepare_prompt(self, messages: list[dict]) -> str:
        """
        Prepare the prompt in the format required by the Gemma-2 model.

        Args:
            messages (List[dict]): A list of messages with roles and content.

        Returns:
            str: The formatted prompt.
        """
        prompt = ""
        for message in messages:
            role = message['role']
            content = message['content']
            if role == 'system':
                prompt += "<start_of_turn>system\n" + content + "<end_of_turn>\n"
            elif role == 'user':
                prompt += "<start_of_turn>user\n" + content + "<end_of_turn>\n"
            elif role == 'model':
                prompt += "<start_of_turn>model\n" + content
        return prompt

    def generate_response(self, prompt: str, max_new_tokens: int = 200) -> str:
        """
        Generate a response from the LLM model based on the given prompt.

        Args:
            prompt (str): The input prompt.
            max_new_tokens (int): The maximum number of new tokens to generate.

        Returns:
            str: The generated response.
        """
        input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        output = self.tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
        return output.strip()

    def process_message(self, text: str, system_prompt: str) -> str:
        """
        Process a message to generate a response from the LLM model.

        Args:
            text (str): The user input text.
            system_prompt (str): The system prompt.

        Returns:
            str: The generated response.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]

        prompt = self.prepare_prompt(messages)
        return self.generate_response(prompt)

    class PipelineManager:
        def __init__(self, cluster_pipe: ClusterPipe, llm: LLM):
            """
            Initialize the PipelineManager with paths to the MTR and labels datasets, and the model names.

            Args:
                llm (LLM)
                cluster_pipe (ClusterPipe)
            """
            self.cluster_pipe = cluster_pipe
            self.llm = llm

        def get_cluster(self, cluster_num: int, okpd_identifier: str) -> pl.DataFrame:
            return self.cluster_pipe.sample_cluster(cluster_num, okpd_identifier)

        def get_cluster_name(self, cluster: pl.DataFrame) -> str:
            system_promt_cluster_name = """Ты русскоязычный ассистент. Ты помогаешь придумывать название для группы товаров"""
            user_promt_cluster_name = """Придуймай обобщающее название для группы товаров.
            ИНСТРУКЦИИ:
            1. Название должно быть четким и напрямую связанным с группой товаров.
            2. Формат названия: существительное (существительное/прилагательное).
            3. Верни только название в описанном формате.
            ВХОДНЫЕ ДАННЫЕ:
            Наименования единиц товаров входящих в группу. Каждое наименование написано с новой строчки.
            Наименования товаров:
            {}.
            ФОРМАТЫ ВЫВОДА:
            существительное
            существительное прилагательное
            существительное (прилагательные)
            ПРИМЕРЫ ВЫВОДА:
            шланги (садовые)
            камеры ночные
            """

            products = cluster["Наименование"].sample(fraction=1).to_list()[:16]
            cluster_name = \
                self.llm.process_message(system_promt_cluster_name, user_promt_cluster_name.format(products)).split(
                    "\n")[1]
            return cluster_name

        def gen_json_str_from_params_list(self, params_list: list[str]) -> str:
            """
            Generates a JSON string from a list of parameter names.

            Args:
            params_list (list of str): List of parameter names.

            Returns:
            str: JSON string with parameters and their corresponding values.
            """
            params_dict = {param: f"value{i + 1}" for i, param in enumerate(params_list)}
            json_str = json.dumps(params_dict, ensure_ascii=False, indent=4)

            return json_str

        def parse_product_properties(self, row: dict, cluster_properties: str, cluster_name: str) -> str:
            system_promt_extract_params_from = """Ты — Сайга, русскоязычный ассистент. Ты извлекаешь свойства товаров из описания параметров."""
            user_promt_struct_params = """Извлеки свойства товара из описания параметров товара следующей группы {}.

            ИНСТРУКЦИИ:
            1. Извлекай параметры из описания в соответствии с определенной структурой. Данная структура представляет собой набор параметров в формате json.
            2. Для каждого параметра из структуры найди соответсвующее ему значение в описании.
            3. Если ты не можешь найти значение параметра в описании, то заполняй этот параметр значением NODATA, но только при действительном отсутствии параметра.
            4. Выводи только json файл с точно такой же структурой и заполненными значениями параметров.

            СТРУКТУРА ПАРАМЕТРОВ И СТРУКТУРА ОТВЕТА:
            json
            {}

            Заполни ее значениями параметров из следующего текста.

            ОПИСАНИЕ ПАРАМЕТРОВ ТОВАРА:
            {}
            """

            json_str_from_params_list = self.gen_json_str_from_params_list(
                cluster_properties.split(";")
            )

            params_raw = f"{row['Наименование']} {row['Маркировка']}: {row['Параметры']}"

            params_parsed = self.llm.process_message(
                user_promt_struct_params.format(cluster_name, json_str_from_params_list, params_raw),
                system_promt_extract_params_from
            )
            start_index = params_parsed.find('{')
            end_index = params_parsed.rfind('}') + 1
            cleaned_json_string = params_parsed[start_index:end_index]
            return cleaned_json_string

        def get_cluster_properties(self,
                                   cluster: pl.DataFrame,
                                   cluster_name: str) -> str:
            system_promt_group_parameters = """Ты — Сайга, русскоязычный ассистент. Ты помогаешь придумывать набор параметров для описания группы товаров"""
            user_promt_group_parameters = """Выдели из описаний набор параметров, которые позволят единым образом описать товары из группы с названием "{}".

            ИНСТРУКЦИИ:
            1. Каждый параметр должен характеризоваться 1 словом.
            2. Набор параметров должен состоять не более чем из 10 параметров. 
            3. Параметры должны основываться исключительно на информации из описаний.
            4. Если описания короткие и неинформативные, ты можешь вернуть менее, чем 10 параметров.
            5. Старайся понять, какие параметры отражают предоставленные описания товаров.
            6. Обращай внимание на цифры и предполагай, какой параметр они могут означать в данном контексте.
            7. Делай параметры разнообразными и не дублирующими друг друга по смыслу.
            8. Возвращай набор параметров как название каждого отдельного параметра с ; в качестве разделителя между ними.
            9. Верни только набор параметров.

            ВХОДНЫЕ ДАННЫЕ (ОПИСАНИЯ ТОВАРОВ):
            Наименования и описания единиц товаров входящих в группу. Каждая пара будет начинаться с новой строчки и представлена в формате наименование товара: описание товара.
            {}

            ФОРМАТ ВЫВОДА:
            параметр 1; параметр 2; параметр 3; параметр n

            ПРИМЕР ВЫВОДА:
            длина; ширина; высота; цвет
            """
            if len(cluster) > 2:
                products = cluster.sample(fraction=1).sample(len(cluster) // 2)
            else:
                products = cluster.head(1)

            products = "\n".join([
                name + " " + params + ": " + mark for name, params, mark in zip(
                    products["Наименование"].to_numpy(),
                    products["Маркировка"].to_numpy(),
                    products["Параметры"].to_numpy(),
                )
            ])

            properties = self.llm.process_message(
                user_promt_group_parameters.format(cluster_name, products), system_promt_group_parameters
            ).split("\n")[1]

            return properties

        def cluster_process(self, cluster_sample: pl.DataFrame) -> pl.DataFrame:
            """
            Process a cluster sample to generate cluster name, properties, and parse product properties.

            Args:
                cluster_sample (pl.DataFrame): DataFrame containing the cluster data.

            Returns:
                pl.DataFrame: DataFrame with added cluster name, properties, and parsed item properties.
            """
            cluster_name = self.get_cluster_name(cluster_sample)
            cluster_prop = self.get_cluster_properties(cluster_sample, cluster_name)

            answer = cluster_sample.with_columns(
                pl.lit(cluster_name).alias("cluster_name"),
                pl.lit(cluster_prop).alias("cluster_properties")
            )

            items_jsons = []
            for item in answer.iter_rows(named=True):
                item_prop = self.parse_product_properties(item, cluster_prop, cluster_name)
                items_jsons.append(item_prop)

            group = answer.with_columns(
                pl.Series(items_jsons).alias("parsed_item_properties")
            )

            products = []
            for row in group.iter_rows(named=True):
                product = {}
                product["код СКМТР"] = row["код СКМТР"]
                product["Наименование"] = row["Наименование"]
                product["Маркировка"] = row["Маркировка"]
                product["Группа"] = row["cluster_name"]
                product["ОКПД2"] = row["ОКПД2"]
                product.update(json.loads(row["parsed_item_properties"]))
                products.append(product)

            return pl.DataFrame(products).fill_null("NODATA")

        def process_okpd(self, okpd_ident: str) -> pl.DataFrame:
            """
            Process all clusters for a given OKPD2 identifier.

            Args:
                okpd_ident (str): The OKPD2 identifier to process.

            Returns:
                pl.DataFrame: DataFrame containing the processed clusters for the given OKPD2 identifier.
            """
            okpd_df = self.cluster_pipe.full_data_clustered.filter(
                pl.col("ОКПД2") == okpd_ident
            )

            unique_clusters = okpd_df["cluster"].unique()

            for c, cluster_id in enumerate(unique_clusters):
                cluster_sample = self.cluster_pipe.sample_cluster(cluster_id, okpd_ident)
                processed_cluster = self.cluster_process(cluster_sample)

                if c == 0:
                    full_data = processed_cluster
                else:
                    full_data = pl.concat([full_data, processed_cluster], how="vertical")

            return full_data
