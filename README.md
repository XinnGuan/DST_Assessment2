# Sentiment Analysis NLP Assessment for Data Science Toolbox

### Comment on Markdown reflections:

The PDF versions of the reflections are created using:

```{sh}
pandoc -o RachelR_Reflection.pdf RachelR_Reflection.md 
pandoc -o PeterP_Reflection.pdf PeterP_Reflection.md 
```

Markdown is an acceptable format, though PDF looks nicer.

### Comment on Report formats:

It is completely fine to present a well commented Rmd or ipynb file. You are welcome to try to generate a beautiful PDF in which all of the results are knitted together, but it can be awkward if content is fundamentally separated. Yes, you can create a PDF from each file and merge the PDF, and doing so once is educational, but it isn't the point of DST.

**Please commit your final output**. It is generally considered bad practice to commit transient content to your repository. This would include the Jupyter Notebook with all of the content competed, and the html output of Rmd. However, for the purposes of generating a one-off assessed report, it is safest to do this, though best only for your final commit. 

This is because it is possible that I cannot run your code, for a good reason or a bad, and therefore I want to see what the output should be.

Why is transient content bad? You repository will get bigger and take longer to process as the whole history of everything that you've generated is stored. Text files compress very nicely for this content, but binary objects such as images and data, hidden inside html or ipynb files, compress badly.

### Comment on data:

Don't commit very large datasets to GitHub, and don't commit modestly large ones unless necessary (and try not to duplicate them). There are file size limits, but it is inefficient. Try to use a different data sharing solution, such as OneDrive, for such data.

### Comment on requirements:

I have explained the mechanisms used for library dependencies within each file. You should *not* need to actually install the dependencies listed below as they are all quite standard; however the proceure is standard and helpful.

# README.md

## Project Group

* Xin Guan
* Emelia Osborne
* Erin Pollard
* Zhihui Zhang

This project has a 25/25/25/25 equity split betwen the four project partners.

## Reading order and requirements

All report content is in the directory:

* report/

The report takes the following structure:

* 01-Introduction.ipynb
* Data
* 02-DataCleaning.ipynb
* 03-LDAModel.ipynb
* 04-Word2vecModel.ipynb
* 05-Classification.ipynb
* 06-Comparison.ipynb
* 07-Discussion.ipynb

### Requirements:
Requirements for the Python code is given in `requirements.txt`; to install, in a virtual environment run:

```{sh}
pip3 install -r requirements.txt
```

## Evidence

Our working is shown in our own directories,

Zhihui performed the data cleaning, Erin & Emelia worked on the LDA model, Xin produced the Word2Vec model and we collectively created the classifciation model and wrote the comparison/discussion together after examination of the results.


Thanks!
