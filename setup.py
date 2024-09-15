from setuptools import setup, find_packages

setup(
    name='light_retrieval_flow',
    version='0.0.1',
    description='Light Retrieval Flow',
    author='Aleksandr Dzhumurat',
    author_email='adzhumurat@yandex.ru',
    package_dir={'': 'src'},  # Tells setuptools to look for packages in the 'src' directory
    packages=find_packages(where='src'),  # Finds all packages in the 'src' directory
    include_package_data=True,  # Includes files specified in MANIFEST.in
    install_requires=[  # List your project dependencies here
        'arrow==1.3.0',
        'boto3==1.34.113',
        'botocore==1.34.113',
        'catboost==1.2.5',
        'elastic-transport==8.13.0',
        'elasticsearch==8.13.2',
        'nltk==3.8.1',
        'numpy==1.26.4',
        'pandas==2.2.2',
        'psycopg2-binary==2.9.9',
        'python-dotenv==1.0.1',
        'PyYAML==6.0.1',
        'requests==2.32.2',
        'scikit-learn==1.5.0',
        'scipy==1.13.1',
        'tqdm==4.66.4',
    ],
)
