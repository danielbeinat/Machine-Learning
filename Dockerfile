FROM continuumio/anaconda3

# Copiar del directorio local (.) al directorio /usr/app del contenedor
#COPY app.py /usr/app
#COPY movies2.csv /usr/app


COPY . /usr/app

# Habilitar puerto
EXPOSE 5000

# Directorio de trabajo del contenedor
WORKDIR /usr/app

# Instalar requerimientos
RUN pip install -r requirements.txt

# Ejecución
CMD python app.py
