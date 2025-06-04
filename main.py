from abc import ABC, abstractmethod
import math
import pandas as pd
import vk_api
import time
from random import randint
from ultralytics import YOLO
import cv2
import numpy as np
import os
import wget
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from typing import List
from catboost import CatBoostClassifier, Pool


# Интерфейс парсера
class Parser(ABC):
    @abstractmethod
    def __get_user(self, id):
        pass

    @abstractmethod
    def parsing_data(self, users_id):
        pass


# Интерфейс классификатора
class Classificator(ABC):
    @abstractmethod
    def __learn_model(self):
        pass

    @abstractmethod
    def classification_data(self):
        pass


# Реализации
class ProfileParser(Parser):
    def __init__(self, access_token: str):
        # Инициализация сессии VK API
        self.vk_session = vk_api.VkApi(token=access_token)
        self.vk = vk_session.get_api()

    def __get_user(self, id):
        try:
            # Запрос информации о пользователе
            user_info = vk.users.get(
                user_ids=id,  # ID или короткое имя пользователя
                fields="bdate,status,career,city,id,domain,"
                "education,schools,universities,"
                "connections,contacts,relation,"
                "counters,personal,about,activities,"
                "books,games,interests,movies,music,quotes,tv,"
                "can_post,can_write_private_message",
            )
            return user_info
        except vk_api.exceptions.ApiError as e:
            print(f"Ошибка VK API: {e}")
            return None

    def parsing_data(self, users_id: list):
        data_df = pd.DataFrame()

        for user in users_id:
            user_data = __get_user(user)

            # получение статистики и флагов
            result_df = pd.DataFrame()

            result_df.loc[0, "id"] = user_data[0].get("id")

            result_df.loc[0, "friends"] = (
                user_data[0].get("counters", {}).get("friends")
            )
            result_df.loc[0, "photos"] = user_data[0].get("counters", {}).get("photos")
            result_df.loc[0, "subscribes"] = (
                user_data[0].get("counters", {}).get("followers")
            )
            result_df.loc[0, "count_albums"] = (
                user_data[0].get("counters", {}).get("albums")
            )
            result_df.loc[0, "count_video"] = (
                user_data[0].get("counters", {}).get("videos")
            )
            result_df.loc[0, "musics"] = user_data[0].get("counters", {}).get("audios")
            result_df.loc[0, "groups"] = user_data[0].get("counters", {}).get("groups")

            # категориальные признаки с несколькими вариантами
            # жизненная позиция
            result_df.loc[0, "worldview"] = (
                user_data[0].get("personal", {}).get("religion")
            )
            result_df.loc[0, "has_worldview"] = (
                True
                if user_data[0].get("personal", {}).get("religion") is not None
                else False
            )
            result_df.loc[0, "important_thing"] = (
                user_data[0].get("personal", {}).get("life_main")
            )
            result_df.loc[0, "has_important_thing"] = (
                True
                if user_data[0].get("personal", {}).get("life_main") is not None
                else False
            )
            result_df.loc[0, "important_in_people"] = (
                user_data[0].get("personal", {}).get("people_main")
            )
            result_df.loc[0, "has_important_in_people"] = (
                True
                if user_data[0].get("personal", {}).get("people_main") is not None
                else False
            )
            result_df.loc[0, "smoking"] = (
                user_data[0].get("personal", {}).get("smoking")
            )
            result_df.loc[0, "has_smoking"] = (
                True
                if user_data[0].get("personal", {}).get("smoking") is not None
                else False
            )
            result_df.loc[0, "alcohol"] = (
                user_data[0].get("personal", {}).get("alcohol")
            )
            result_df.loc[0, "has_alcohol"] = (
                True
                if user_data[0].get("personal", {}).get("alcohol") is not None
                else False
            )
            result_df.loc[0, "political_view"] = (
                user_data[0].get("personal", {}).get("political")
            )
            result_df.loc[0, "has_political_view"] = (
                True
                if user_data[0].get("personal", {}).get("political") is not None
                else False
            )
            result_df.loc[0, "has_inspiration"] = (
                True
                if user_data[0].get("personal", {}).get("inspired_by") is not None
                else False
            )

            marital_status = user_data[0].get("relation", 0)
            result_df.loc[0, "marital_status"] = True if marital_status != 0 else False

            # булевые категориальные

            result_df["has_birthday"] = (
                True if user_data[0].get("bdate") is not None else False
            )
            result_df["has_city"] = (
                True if user_data[0].get("city") is not None else False
            )
            result_df["has_status"] = (
                True if user_data[0].get("status") is not None else False
            )
            result_df["has_custom_link"] = (
                True
                if ("id" + str(user_data[0].get("id", "")))
                != user_data[0].get("domain", "")
                else False
            )

            #
            result_df["has_work"] = True if user_data[0]["career"] else False
            result_df["has_education"] = True if user_data[0]["universities"] else False
            result_df["has_school"] = True if user_data[0]["schools"] else False

            # интересы
            result_df.loc[0, "has_favorite_music"] = (
                True if user_data[0].get("movies") is not None else False
            )
            result_df.loc[0, "has_books"] = (
                True if user_data[0].get("books") is not None else False
            )
            result_df.loc[0, "has_games"] = (
                True if user_data[0].get("games") is not None else False
            )
            result_df.loc[0, "has_quotes"] = (
                True if user_data[0].get("quotes") is not None else False
            )
            result_df.loc[0, "has_TV_shows"] = (
                True if user_data[0].get("tv") is not None else False
            )
            result_df.loc[0, "has_movies"] = (
                True if user_data[0].get("movies") is not None else False
            )
            result_df.loc[0, "has_activity"] = (
                True if user_data[0].get("activities") is not None else False
            )
            result_df.loc[0, "has_information"] = (
                True if user_data[0].get("about") is not None else False
            )

            # возможности
            # открыты ли личные сообщения?
            result_df.loc[0, "chat_is_open"] = user_data[0].get(
                "can_write_private_message"
            )
            # открыта ли стена для записей МОЖНО ПОСТИТЬ НА СТЕНУ
            result_df.loc[0, "profile_wall_is_open"] = user_data[0].get("can_post")

            # рассчет среднего кол-ва лайков и комментариев на фото
            avg_photo_likes = 0
            photo_likes = 0

            for i in range(0, (int(result_df.at[0, "photos"]) // 200) + 1):
                offset = i * 200
                count = offset + 200  # максимальное количество фото за один запрос

                response = vk.photos.getAll(
                    owner_id=user_data[0]["id"],
                    extended=1,
                    offset=offset,
                    count=count,
                    photo_sizes=0,
                )

                time.sleep(randint(1, 3))

                for photo in response["items"]:
                    photo_likes += photo["likes"]["count"]

            avg_photo_likes = photo_likes / result_df.at[0, "photos"]

            time.sleep(randint(1, 3))

            response = vk.photos.getAlbums(owner_id=user_data[0]["id"], need_system=1)

            avg_photo_comments = 0
            count_coments = 0
            offset = 0
            count = 100  # максимум комментариев за 1 запрос

            for album in response["items"]:
                while True:
                    response_com = vk.photos.getAllComments(
                        owner_id=user_data[0]["id"],
                        album_id=album["id"],
                        offset=offset,
                        count=count,
                    )

                    if not response_com["items"]:
                        break

                    count_coments += response_com["count"]
                    print(count_coments)

                    offset += count

                    time.sleep(randint(1, 3))

                print(album["size"])
                avg_coments = count_coments / album["size"]

                avg_photo_comments += avg_coments

            avg_photo_comments = avg_photo_comments / response["count"]

            result_df.loc[0, "avg_photo_likes"] = avg_photo_likes
            result_df.loc[0, "avg_photo_comments"] = avg_photo_comments

            offset = 0
            count = 100  # максимум постов за 1 запрос

            posts = pd.DataFrame()
            reposts = pd.DataFrame()

            while True:
                response = vk.wall.get(
                    owner_id=user_data[0].get("domain"), offset=offset, count=count
                )

                if not response["items"]:
                    break

                for post in response["items"]:

                    if "copy_history" in post:
                        i = len(reposts.index)

                        reposts.loc[i, "likes"] = post["likes"]["count"]
                        reposts.loc[i, "reposts"] = post["reposts"]["count"]
                        reposts.loc[i, "comments"] = post["comments"]["count"]
                        reposts.loc[i, "views"] = post["views"]["count"]
                    else:
                        i = len(posts.index)

                        posts.loc[i, "likes"] = post["likes"]["count"]
                        posts.loc[i, "reposts"] = post["reposts"]["count"]
                        posts.loc[i, "comments"] = post["comments"]["count"]
                        posts.loc[i, "views"] = post["views"]["count"]

                offset += count

                time.sleep(randint(1, 3))

            result_df.loc[0, "avg_post_likes"] = posts["likes"].mean()
            result_df.loc[0, "avg_post_reposts"] = posts["reposts"].mean()
            result_df.loc[0, "avg_post_comments"] = posts["comments"].mean()
            result_df.loc[0, "avg_post_views"] = posts["views"].mean()
            result_df.loc[0, "posts"] = len(posts.index)

            result_df.loc[0, "avg_repost_likes"] = reposts["likes"].mean()
            result_df.loc[0, "avg_repost_reposts"] = reposts["reposts"].mean()
            result_df.loc[0, "avg_repost_comments"] = reposts["comments"].mean()
            result_df.loc[0, "avg_repost_views"] = reposts["views"].mean()
            result_df.loc[0, "reposts"] = len(reposts.index)

            data_df = pd.concat([data_df, result_df], ignore_index=True)

        return data_df


# Реализации
class PhotoParser(Parser):
    def __init__(self, access_token: str):
        # Инициализация сессии VK API
        self.vk_session = vk_api.VkApi(token=access_token)
        self.vk = vk_session.get_api()

    def __get_user(self, id):
        try:
            # Запрос информации о пользователе
            user_info = vk.users.get(
                user_ids=id, fields="id,counters"  # ID или короткое имя пользователя
            )
            return user_info
        except vk_api.exceptions.ApiError as e:
            print(f"Ошибка VK API: {e}")
            return None

    def __download(self, user_id, photo_id, url, path):
        user_folder = str(user_id)
        newpath = path + "/" + user_folder
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        newpath = newpath + "/" + str(photo_id) + ".jpg"
        wget.download(url, out=newpath)

    def __find_objects(self, path_to_images, yolo_model):

        # Получаем папки пользователей
        folders = os.listdir(path_to_images)

        # Загрузка модели YOLO
        model = YOLO(yolo_model)

        classes_names = model.names

        columns = ["id", "photo"]

        columns.extend(classes_names.values())

        df = pd.DataFrame(columns=columns)

        for user in folders:

            user_path = path_to_images + "/" + user

            # Получаем список фото в папке
            photo_files = os.listdir(user_path)

            for photo in photo_files:

                image = cv2.imread(user_path + "/" + photo)

                results = model(image)[0]

                a = results.boxes.cls.cpu().numpy().astype(np.int32)
                unique, counts = np.unique(a, return_counts=True)
                photo_category = dict(zip(unique, counts))

                data_row = [user, photo]

                for coco_category in classes_names:
                    value = photo_category.get(coco_category, 0)
                    data_row.append(value)

                df.loc[len(df)] = data_row

        return df

    def parsing_data(self, users_id: list, path: str, yolo_model):
        data_df = pd.DataFrame()

        for user in users_id:
            user_data = __get_user(user)
            offset = 0
            count = 200
            i = len(user_data.index)

            while True:
                response = vk.photos.getAll(
                    owner_id=user_data[0]["id"],
                    extended=1,
                    offset=offset,
                    count=count,
                    photo_sizes=0,
                )

                if not response["items"]:
                    break

                for photo in response["items"]:
                    __download(
                        user_data[0]["id"],
                        photo["id"],
                        photo["orig_photo"]["url"],
                        path,
                    )

                    time.sleep(0.5)

                time.sleep(randint(1, 3))

            data_df.loc[i, "id"] = user["id"]
            data_df.loc[i, "photos"] = user.get("counters", {}).get("photos")

        photo_df = __find_objects(path, yolo_model)

        # количество фото с каждым из объектов
        temp_df_photo_features = photo_df.copy()

        for column in photo_df.columns[2:82]:
            temp_df_photo_features[column] = np.where(photo_df[column] > 0, 1, 0)

        result_temp_df_photo_features = (
            temp_df_photo_features.groupby(by="id")[photo_df.columns[2:82]]
            .sum()
            .reset_index()
        )

        # проценты от общего кол-ва фото
        temp_df_photo_features = data_df.merge(
            result_temp_df_photo_features, left_on="id", right_on="id", how="left"
        )

        for column in temp_df_photo_features.columns[2:]:
            temp_df_photo_features[column] = temp_df_photo_features.apply(
                lambda x: x[column] / x["photos"] if x[column] != None else None, axis=1
            )

        temp_df_photo_features = temp_df_photo_features.drop(columns="photos")

        return temp_df_photo_features


# Реализации
class MusicParser(Parser):
    def __init__(self, access_token: str):
        # Инициализация сессии VK API
        self.vk_session = vk_api.VkApi(token=access_token)
        self.vk = vk_session.get_api()

    def __get_user(self, id):
        try:
            # Запрос информации о пользователе
            user_info = vk.users.get(
                user_ids=id, fields="id,counters"  # ID или короткое имя пользователя
            )
            return user_info
        except vk_api.exceptions.ApiError as e:
            print(f"Ошибка VK API: {e}")
            return None

    def __download(self, user_id, photo_id, url, path):
        user_folder = str(user_id)
        newpath = path + "/" + user_folder
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        newpath = newpath + "/" + str(photo_id) + ".jpg"
        wget.download(url, out=newpath)

    def __get_audio_features(self, data_audio, sr):

        features = []

        audio_file, _ = librosa.effects.trim(data_audio)

        # chroma stft
        hop_length = 5000
        chromagram = librosa.feature.chroma_stft(
            y=audio_file, sr=sr, hop_length=hop_length
        )
        chroma_stft_mean = np.mean(chromagram)
        chroma_stft_var = np.var(chromagram)
        features.append(chroma_stft_mean)
        features.append(chroma_stft_var)

        # rms
        rms = librosa.feature.rms(y=audio_file)
        rms_mean = np.mean(rms)
        rms_var = np.var(rms)
        features.append(rms_mean)
        features.append(rms_var)

        # Spectral Centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_file, sr=sr)[0]
        spectral_centroid_mean = np.mean(spectral_centroids)
        spectral_centroid_var = np.var(spectral_centroids)
        features.append(spectral_centroid_mean)
        features.append(spectral_centroid_var)

        # spectral_bandwidth
        spec_bw = librosa.feature.spectral_bandwidth(y=audio_file, sr=sr)
        spectral_bandwidth_mean = np.mean(spec_bw)
        spectral_bandwidth_var = np.var(spec_bw)
        features.append(spectral_bandwidth_mean)
        features.append(spectral_bandwidth_var)

        # Spectral Rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_file, sr=sr)[0]
        spectral_rolloff_mean = np.mean(spectral_rolloff)
        spectral_rolloff_var = np.var(spectral_rolloff)
        features.append(spectral_rolloff_mean)
        features.append(spectral_rolloff_var)

        # Zero Crossing Rate
        zero_crossing_rate = librosa.zero_crossings(audio_file, pad=False)
        zero_crossing_rate_mean = np.mean(zero_crossing_rate)
        zero_crossing_rate_var = np.var(zero_crossing_rate)
        features.append(zero_crossing_rate_mean)
        features.append(zero_crossing_rate_var)

        # Harmonics and Perceptrual
        y_harm, y_perc = librosa.effects.hpss(audio_file)
        harmony_mean = np.mean(y_harm)
        harmony_var = np.var(y_harm)
        perceptr_mean = np.mean(y_perc)
        perceptr_var = np.var(y_perc)
        features.append(harmony_mean)
        features.append(harmony_var)
        features.append(perceptr_mean)
        features.append(perceptr_var)

        # Tempo BMP (beats per minute)
        tempo, _ = librosa.beat.beat_track(y=data_audio, sr=sr)
        features.append(tempo)

        # Mel-Frequency Cepstral Coefficients
        mfccs = librosa.feature.mfcc(y=audio_file, sr=sr)

        for i in range(20):
            value = np.mean(mfccs[i])
            features.append(value)

            value = np.var(mfccs[i])
            features.append(value)

        return features

    def get_audio_genre(self, file, model, scaler_music, columns):

        data, sr = librosa.load(file, sr=22050)

        metka = np.zeros(10)

        parts = len(data) // sr // 30

        if parts > 0:
            for i in range(parts):
                test_feat = __get_audio_features(
                    data[i * sr * 30 : (i + 1) * sr * 30], sr
                )
                testt_df = pd.DataFrame(columns=columns)
                testt_df.loc[len(testt_df)] = test_feat
                testt_df = pd.DataFrame(
                    scaler_music.transform(testt_df), columns=testt_df.columns
                )

                preds = model.predict(testt_df)

                metka[preds[0]] = metka[preds[0]] + 1

            genre = np.argmax(metka)
        else:
            genre = -1

        return genre

    def parsing_data(self, users_id: list, path: str, yolo_model):
        data_df = pd.DataFrame()

        audio_df = pd.read_csv("./gtzan/Data/features_30_sec.csv")
        audio_df = audio_df.drop(columns=["filename", "length"])

        le = LabelEncoder()
        y = le.fit_transform(audio_df["label"].values)
        audio_df = audio_df.drop(columns="label")

        scaler_music = StandardScaler()
        X = pd.DataFrame(
            data=scaler_music.fit_transform(audio_df), columns=audio_df.columns
        )

        get_audio_genre(X.columns)

        # for user in users_id:

        return temp_df_photo_features


class ANNClassificator(Classificator):
    def __init__(self, best_columns: list, h5_file: str):
        self.best_columns = best_columns
        self.model = load_model(h5_file)

    def classification_data(self, data: pd.DataFrame) -> pd.DataFrame:
        temp_df = data[self.best_columns]

        y_pred_proba = self.model.predict(temp_df)
        y_pred = np.argmax(y_pred_proba, axis=1)

        return y_pred


class CatBoostClassificator(Classificator):
    def __init__(self, best_columns: list, file: str):
        self.best_columns = best_columns
        self.model = CatBoostClassifier().load_model(file)

    def classification_data(self, data: pd.DataFrame) -> pd.DataFrame:
        temp_df = data[self.best_columns]

        y_pred = self.model.predict(temp_df)

        return y_pred


class Controller:
    def __init__(self, parsers: List[Parser], classificators: List[Classificator]):
        self.data = pd.DataFrame()
        self.parsers = parsers
        self.classificators = classificators

    def parse_with(self, parser_type: type, users_id: list) -> pd.DataFrame:
        for parser in self.parsers:
            if isinstance(parser, parser_type):
                temp_df = parser.parsing_data(users_id)
                temp_df = temp_df.drop(columns=["id"])
                self.data = temp_df
        raise ValueError(f"Парсер типа {parser_type.__name__} не найден.")

    def parse_all(self, users_id: list):
        data = pd.DataFrame()

        for parser in self.parsers:
            if data.empty:
                data = parser.parsing_data(users_id)
            else:
                temp_df = parser.parsing_data(users_id)
                data = data.merge(temp_df, on="id")

        data = data.drop(columns=["id"])

        self.data = data.copy()

    def set_missing_data(self, dataset: pd.DataFrame, imputer):
        imp = imputer.fit(dataset)
        self.data = pd.DataFrame(imp.transform(self.data), columns=self.data.columns)

    def use_classificator(self, trait: int):
        return self.classificators[trait].classification_data()

    def save_data(self, path: str):
        self.data.to_csv(path, index=False)
