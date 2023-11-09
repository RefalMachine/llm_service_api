# Сервис для восстановления пунктуации и смысловой структуры (параграфов)

Пример команды для сборки Docker контейнера
```
docker build -t struct_recover_server . 
```

Для работы необходимо скачать модели:
https://disk.yandex.ru/d/PgMeMwvV8LMlGg 

После сборки контейнера и распаковки моделей запуск осуществляется командой:
```
docker run -d --rm -it -p HOSTPORT:80 --mount src=PATHTOMODELS,target=/workdir/models,type=bind struct_recover_server
```

где *HOSTPORT* - порт, с которого будет осуществлять доступ (например, 7860), *PATHTOMODELS* - путь к распакованной директории models (например, ./models)