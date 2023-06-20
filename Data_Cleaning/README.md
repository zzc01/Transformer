# Cleaning the German-English language sentence pair data set

The German-English data set is downloaded from http://www.manythings.org/anki/

The [clean_pairs](clean_pairs.ipynb) script does the following clean up steps:

1. Ignore all chars that cannot be represented in ASCII
2. Convert all chars to lowercase
3. Remove punctuations
4. Remove all non-printable chars
5. Remove none alphabet words

This code is refereneced from https://machinelearningmastery.com/

# Cleaning the data set using PySpark on Databricks
As more language pairs are added the data set can grow very large. In this case PySpark can be used to help scale to huge datasets. It has RDDs and dataframes that are powerful to process the datasets on clusters. In [Data_Cleaning_PySpark](data_cleaning_pyspark.ipynb) script we use PySpark to process the dataset. Here is a brief description of the process: 
1. First import the text file into a PySpark dataframe
2. Split the dataframe into two columns one for the English sentences and the second for the German sentences
3. Define an UDF with steps described above to clean up each columns
4. Save the dataframe into a CSV file 







