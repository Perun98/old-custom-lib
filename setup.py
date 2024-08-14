from setuptools import setup, find_packages

setup(
    name="myfunc",
    version="2.0.77a",
    packages=find_packages(),
    install_requires=[
        'aiohttp==3.9.5',
        'azure-core==1.30.2',
        'azure-storage-blob==12.21.0',
        'beautifulsoup4==4.12.3',
        'cohere==5.6.2',
        'html2docx==1.6.0',
        'langchain==0.2.11',
        'langchain-community==0.2.10',
        'langchain-openai==0.1.17',
        'langchain-text-splitters==0.2.2',
        'Markdown==3.6',
        'matplotlib==3.9.1',
        'mysql-connector-python==9.0.0',
        'neo4j==5.22.0',
        'networkx==3.3',
        'openai==1.37.1',
        'pandas==2.2.2',
        'pdfkit==1.0.0',
        'pillow==10.4.0',
        'pinecone==4.0.0',
        'pinecone-text==0.9.0',
        'pydub==0.25.1',
        'pyodbc==5.1.0',
        'pypandoc==1.13',
        'python-docx==1.1.2',
        'PyPDF2==3.0.1',
        'PyYAML==6.0.1',
        'requests==2.32.3',
        'semantic-router==0.0.54',
        'setuptools==71.1.0',
        'sounddevice==0.4.7',
        'soundfile==0.12.1',
        'streamlit==1.37.0',
        'streamlit-authenticator==0.3.2',
        'streamlit-javascript==0.1.5',
        'tiktoken==0.7.0',
        'tqdm==4.66.4',
        'Unidecode==1.3.8',
    ]
)
