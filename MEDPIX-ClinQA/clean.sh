#!/bin/bash

# ollama run llama3.1 "$(cat ./dataset.csv)" "Read every line yourself no need to use any library
# and if there are answers that talk about
# the dimensions that modify the answer so that it avoids talking about it,
# modify answers that talk about demography or geography, remove rows that talk
# about codes like Acr, icd and so on, remove the rows that talk about history,
# if the answer is 60 words or more then summarize it to the essential bits ..
# keep in mind that This dataset will be used for finetuning ; and your output
# should only be csv with the same columns as the input. so no text before and after processing.
# And each row should be in 1 line
# " > out_big.csv

# Step 1: Split the CSV file into chunks of 100 lines each (adjust the number of lines if needed)
split -l 10 --additional-suffix=.csv ./dataset_sa.csv chunk_

# Create the output file and ensure it's empty before appending
echo "" > out_big.csv

# Step 2: Process each chunk using `ollama` and append output to the final file
for chunk in chunk_*.csv; do
		echo "--------------------" >> out_big.csv
		ollama run llama3.1 "$(cat "$chunk")" \
		"Read every line yourself no need to use any library such as python
		and if there are answers that talk about
		the dimensions that modify the answer so that it avoids talking about it,
		modify answers that talk about demography or geography, remove rows that talk
		about codes like Acr, icd and so on, remove the rows that talk about history,
		if the answer is 60 words or more then summarize it to the essential bits ..
		keep in mind that This dataset will be used for finetuning ; and your output
		should only be csv with the same columns as the input. so no text before and after processing.
		strip any new lines if they are present in the answer . feel free to use bullet points •
		so the output should always be image_id , case_id , question , answer ,ans_len, mode , split 
		And each row should be in 1 line. and nothing else
		so just return the processed data as the output is directly going to be piped to a csv" >> out_big.csv  # Append directly to the final file
	echo "--------------------" >> out_big.csv
done

# Step 3: Clean up intermediate files (optional)
rm chunk_*.csv

echo "Processing complete. Output written to out_big.csv"
