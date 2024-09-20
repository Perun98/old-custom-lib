from setuptools import setup, find_packages

setup(
    name="myfunc",
    version="2.0.83c",
    packages=find_packages(),
    install_requires=[
        'aiohttp==3.9.5',
        'azure-core==1.30.2',
        'azure-storage-blob==12.21.0',
        'beautifulsoup4==4.12.3',
        'chardet==5.2.0',
        'cohere==5.6.2',
        'html2docx==1.6.0',
        'langchain',
        'langchain-community',
        'langchain-openai',
        'langchain-text-splitters',
        'Markdown==3.6',
        'matplotlib==3.9.1',
        'mysql-connector-python',
        'neo4j',
        'networkx==3.3',
        'nltk==3.8.1',
        'openai',
        'pandas',
        'pdfkit==1.0.0',
        'pillow==10.4.0',
        'pinecone',
        'pinecone-text',
        'pydub==0.25.1',
        'pyodbc==5.1.0',
        'pypandoc==1.13',
        'python-docx==1.1.2',
        # 'python-magic-bin',
        'PyPDF2==3.0.1',
        'PyYAML==6.0.1',
        'requests==2.32.3',
        'semantic-router==0.0.54',
        'setuptools==71.1.0',
        'sounddevice==0.4.7',
        'soundfile==0.12.1',
        'streamlit',
        'streamlit-authenticator',
        'streamlit-javascript',
        'tiktoken==0.7.0',
        'tqdm==4.66.4',
        'Unidecode==1.3.8',
    ]
)
