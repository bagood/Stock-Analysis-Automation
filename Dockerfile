FROM mcr.microsoft.com/dotnet/sdk:8.0

USER root
RUN apt-get update && \
    apt-get install -y python3.10 python3-pip && \ 
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install -r requirements.txt --break-system-packages

RUN dotnet workload update

USER app
WORKDIR /app

ENV PATH="$PATH:/home/app/.dotnet/tools"

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--LabApp.token=''"]