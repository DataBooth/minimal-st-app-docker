<!-- date: 2017-07-11T23:02:45+00:00
url: /2017/07/11/did-a-male-octogenarian-really-survive-the-sinking-of-the-rms-titanic-2/
featured_image: /wp-content/uploads/2017/06/titanic2-e1498922251343.jpg -->

# Did a male octogenarian really survive the sinking of the rms titanic
## Or: Is there a long-standing error in an oft used dataset?

As it’s not necessarily a word we use often, let me paraphrase: did an 80 year old guy really manage to make it out of the freezing waters to safety following the infamous maritime disaster?

The short answer is NO. However, read on and let me explain how this article came to be as part of my Data Science travels – in a <a href="http://www.kaggle.com" target="_blank" rel="noopener">Kaggle</a> warm up “competition” specifically.

### Machine Learning from Disaster (Kaggle Dataset) {.competition-header__title}

To set the scene, Kaggle has a challenge to determine what kinds of people were likely to survive. As a quick aside, I think the Kaggle crowdsourcing approach to solving problems in data science is awesome – as confirmed by them recently passing one million members. Their competition asks participants to apply the tools of machine learning to predict which passengers survived the tragedy. The Kaggle datasets are located at <a href="https://www.kaggle.com/c/titanic/data" target="_blank" rel="noopener">https://www.kaggle.com/c/titanic/data</a>. There are two files of interest to us: train.csv and test.csv.

These datasets are used widely within Data Science introductions and examples – almost representing the “<a href="https://en.wikipedia.org/wiki/%22Hello,_World!%22_program" target="_blank" rel="noopener">Hello World</a>” of Data Science 101, something akin to the famous <a href="https://en.wikipedia.org/wiki/Iris_flower_data_set" target="_blank" rel="noopener">Iris flower dataset</a>. Some examples where the Kaggle Titanic datasets have been used include:

  * <a href="https://channel9.msdn.com/Blogs/raw-tech/Using-Azure-Machine-Learning-to-Predict-Who-Will-Survive-the-Titanic-Part-1" target="_blank" rel="noopener">Azure ML</a>
  * <a href="https://www.ibm.com/developerworks/library/ba-bluemix-trs-predictive-analytics-with-dashdb/index.html" target="_blank" rel="noopener">IBM Bluemix</a>
  * <a href="https://rapidminer.com/resource/rapidminer-advanced-analytics-demonstration/" target="_blank" rel="noopener">Rapid Miner</a>
  * <a href="https://public.tableau.com/en-us/s/search/all/titanic" target="_blank" rel="noopener">Tableau</a>
  * <a href="https://www.udemy.com/python-for-data-science-and-machine-learning-bootcamp/learn/v4/content" target="_blank" rel="noopener">Udemy: Python for Data Science and Machine Learning Bootcamp online course</a>

There are plenty of other examples out there. In other words, this dataset has or should have received a very significant amount of scrutiny. However, maybe the depth and quality of this scrutiny has been limited. Perhaps this dataset is one of the most frequently downloaded datasets on the Internet – among aspiring data scientists at least!

### Let’s go back to Titanic with Python (and Pandas)

(Apologies for the movie allusion.)

So I started by taking a look at the two datasets (train.csv and test.csv). Actually most of my attention was focussed on the training dataset; after all, that’s what we use to “train” our model. Of course, a model, any model, is dependent upon the quality of the data. After all, it is **Data** Science we are doing.

So I loaded up the training dataset (train.csv) into my Jupyter notebook and just looked at a quick summary of the data. Besides import statements, it is just one simple command:

<pre>pd.read_csv('train.csv').describe()</pre>

The output is as follows:

<img loading="lazy" class=" size-full wp-image-169 aligncenter" src="https://i2.wp.com/mjboothaus.wpcomstaging.com/wp-content/uploads/2017/06/screenshot-2017-07-02-00-31-02.png?resize=720%2C255&#038;ssl=1" alt="Screenshot 2017-07-02 00.31.02" width="720" height="255" data-recalc-dims="1" /> 

The first thing that grabbed my attention was the maximum age of 80 for a passenger. I have to admit I was surprised by such an elderly person being aboard (remember we are in 1912 when the average life expectancy was somewhere between 50 and 55). So I thought I’d just dig a little further, look at the distribution of ages and then see who this 80 year old person(s) was. The age distribution is bimodal, with a bump for the kids and then again around the median age of 28.

<img loading="lazy" class=" size-full wp-image-172 aligncenter" src="https://i1.wp.com/mjboothaus.wpcomstaging.com/wp-content/uploads/2017/06/screenshot-2017-07-02-00-40-57.png?resize=505%2C302&#038;ssl=1" alt="Screenshot 2017-07-02 00.40.57" width="505" height="302" data-recalc-dims="1" /> 

So who is our octogenarian?

### Algernon Henry Wilson Barkworth

It turns out that his name was <a href="https://www.encyclopedia-titanica.org/titanic-survivor/algernon-barkworth.html" target="_blank" rel="noopener">Algernon Henry Wilson Barkworth</a>. This gentleman was indeed 80 years old when he died. However, he died on <a href="http://www.titanic-titanic.com/algernon_henry_wilson_barkworth.shtml" target="_blank" rel="noopener">7 January 1945</a> in the UK, about 32 years after the Titanic sank in 1912. He was 47-48 (depending on the data source you use) at the time of the accident in 1912. I<span class="fact">n passing, according to the <a href="http://www.titanicfacts.net/titanic-survivors.html" target="_blank" rel="noopener">Titanic Facts website</a>, </span>the oldest survivor overall was Mrs. Mary Eliza Compton (First Class) at 64 years, 8 months and 8 days.

So it seems that this is an error in the dataset. We all make mistakes, but are these errors mistakes or <a href="https://en.wikipedia.org/wiki/Easter_egg_(media)" target="_blank" rel="noopener">easter eggs</a>? Unfortunately, the source(s) of the Kaggle datasets for the Titanic exercise appear not to be provided. Initially I speculated that the good folk at Kaggle had intentionally introduced some errors into the dataset (and had a secret USD $1m award) as part of the exercise, to remind people that much of data science is about wrangling, cleaning and verifying the data; the cool modelling bit only happens after these tasks are (mostly) complete.

However, after doing some non-trivial research it would appear that the Kaggle datasets are derived almost certainly from the [titanic3.csv][1] file. There is a description of the dataset provided in the accompanying <a href="http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3info.txt" target="_blank" rel="noopener">information file</a> which also discusses its origins. According to this file, these datasets reflected the state of data available as of 2 August 1999. Comparing this file with the union of train.csv and test.csv (after dropping some additional columns namely ‘boat’, ‘body’ and ‘home.dest’ from the titanic3 dataset) reveals they are identical.

### So, are there other errors for ages in the dataset?

So given I found this, I thought I should take a closer look at the dataset and see if there was some systematic error. So I began by looking at another supposedly elderly passenger: <a href="https://www.encyclopedia-titanica.org/titanic-survivor/julia-florence-cavendish.html" target="_blank" rel="noopener">Julia Florence Cavendish</a> (who also survived). Yes — another one! She was 25 at the time of the disaster. She died age 76 (around the 50th anniversary of the Titanic’s demise). Of course, there are now no survivors from the Titanic disaster, everyone has died. Even Milvina Dean, the last living survivor, died on <span class="fact">31 May 2009 at the </span>age of 97 (she was 9 weeks old at the time of the disaster). Ultimately working with this dataset is a very sobering experience and there are many sad stories. For example, the 11 members of the Sage family (Third class) who all perished.

There are plenty more (significant) discrepancies with respect to age in the data set. For the interested reader, I have placed an <a href="https://github.com/Mjboothaus/blog/tree/master/titanic" target="_blank" rel="noopener">Excel workbook on GitHub</a> which contains some of my reconciliation of the age data, comparing the data in the Kaggle datasets against the tables from TitanicFacts.net. The figure below (produced by the Jupyter notebook – see the Appendix for details) illustrates the distribution count of the differences in reported age. Note it is truncated at 30 for this figure as many values have a small difference in the reported ages. However, there are 143 passengers that have a difference of greater than two years.

<img loading="lazy" class=" size-full wp-image-513 aligncenter" src="https://i2.wp.com/mjboothaus.wpcomstaging.com/wp-content/uploads/2017/06/titanic_agediff.png?resize=367%2C265&#038;ssl=1" alt="Titanic_AgeDiff" width="367" height="265" data-recalc-dims="1" /> 

I think it is reasonable to assume that the majority of the age column in the dataset may be one to two years out. Given the lack of accurate data for the dates of birth of many of the passengers, their ages can easily be out by at least a year.

Are there any other errors in the dataset? On first inspection, the passenger class field also appears to have some errors. However, as yet I have not conducted a complete reconciliation as it is somewhat time-consuming.

### Have others noticed these errors?

I have been wondering if others have noticed this error. As at 5:21 pm AEST on 2 July 2017 there were 6,961 registered teams for this Kaggle “Getting Started” competition which has been running for over 5 years. I suspect there are significantly more individuals who have not registered but have investigated the data and attempted some solutions. From the research I did I was unable to find anyone else that had found it and felt it worthy of reporting. Maybe attention to detail is not that important in an age of “Big” data. I tend to think it is important and it is certainly one of my skills that has been a faithful friend.

As an example, from a <a href="https://www.kaggle.com/headsortails/pytanic/" target="_blank" rel="noopener">notebook posted on Kaggle</a> the following comments are made with respect to some analysis of the Age variable:

> _Age:_ The medians are identical. However, it’s noticeable that fewer young adults have survived (ages 18 – 30-ish) whereas children younger than 10-ish had a better survival rate. **Also, there are no obvious outliers that would indicate problematic input data.** The highest ages are well consistent with the overall distribution.

I am sure that the author of the notebook is well-intentioned in their comment, however, it demonstrates that attention to detail is not necessarily a widely shared ability. In passing, that’s why I think being part of a diverse team is essential to doing great Data Science.

If we regard the collection of data as an experiment (presumably it’s not called Data **Science** by accident). Then any experiment has errors associated with it which we need to allow for. For instance, in the current example, it is clear that there are errors associated with the reported ages (in 1912). Therefore, assuming we want to still include this age variable in our analysis, how do we allow for these errors? Intuitively we don’t want to fit “too closely”  to the data given an input that has some degree of inherent uncertainty. In other cases, the data has essentially been corrupted and needs to be fixed prior to any analysis (e.g. a person who was 47/8 at the time and not 80; or who was a passenger travelling in first class, not second class).

Machine Learning under uncertainty is something that I am very interested in. How should we take uncertainty into account in the ways algorithms work? How should we reconfigure out data collection procedures to minimise errors? Anyway, I will post on this another time when I’ve had time to establish the current state of play in this domain.

### So is there any absolute (Titanic) truth out there?

Yes, I liked to watch the <a href="https://en.wikipedia.org/wiki/The_X-Files" target="_blank" rel="noopener">X-Files</a> back in the mid to late 1990s. Nonetheless, verifying the veracity of your data is essential. Two wrongs do not make a right and typically lead to further errors. For this post, I have relied upon the following websites for passenger list data,

  * <a href="http://www.titanic-titanic.com/index.shtml" target="_blank" rel="noopener">Titanic-Titanic.com</a>
  * <a href="https://www.encyclopedia-titanica.org" target="_blank" rel="noopener">Encyclopedia Titanica</a>
  * <a href="http://www.titanicfacts.net" target="_blank" rel="noopener">Titanic Facts</a>

however, I have not taken the time to establish if their data is beyond reproach.

Encyclopedia Titanica seems to be regarded as the most definitive source for information about the Titanic passengers. The data for the Wikipedia page which discusses the <a href="https://en.wikipedia.org/wiki/Passengers_of_the_RMS_Titanic" target="_blank" rel="noopener">Passengers of the RMS Titanic</a> is based on Encyclopaedia Titanica. As such it has Mr. Barkworth’s age in 1912 as being 47. Please note that I am not implying that any or all of the sources are not subject to errors themselves, however, it would seem that the train and test datasets could be “enriched” by a systematic comparison with the Encyclopaedia Titanica data.

### Conclusion: what is the impact of these errors?

This will be the subject of a subsequent post which I am working on. From what I have observed so far, age is not one of the primary drivers behind the prediction of survived/perished (it seems to be the 4th most influential factor). However, given that these do not appear to be well-defined systematic errors, it is difficult to say upfront what the overall impact will be.

Irrespective of the impact, I hope this post serves as a reminder to us to be humble as one approaches any problem. Also, to ask good questions and to rigorously interrogate your data and to avoid making assumptions. I worked as an Operational Risk professional for some time and every day was a reminder of human fallibility, no matter how diligent or well-intentioned we are in executing our roles. In an age of &#8220;Big&#8221; data, this is a helpful reminder that even with &#8220;Small&#8221; data it is easy to miss the wood for the trees.

As a courtesy, and as part of finalising this post, I have <a href="https://www.kaggle.com/contact" target="_blank" rel="noopener">contacted Kaggle</a> to highlight these issues with their warm-up competition dataset.

### References

Links to the primary sources that I used are listed below:

  1. <a href="https://www.kaggle.com/c/titanic" target="_blank" rel="noopener">Titanic: Machine Learning from Disaster (Kaggle – Getting Started Prediction Competition)</a>
  2. <a href="http://www.titanic-titanic.com/index.shtml" target="_blank" rel="noopener">Titanic-Titanic.com</a>
  3. <a href="https://www.encyclopedia-titanica.org" target="_blank" rel="noopener">Encyclopedia Titanica</a>
  4. <a href="http://www.titanicfacts.net" target="_blank" rel="noopener">Titanic Facts</a>

## 

## Appendices

### Jupyter Notebook

I am a great believer in the reproducibility of research/analysis (as well all make mistakes) so the simple Python data manipulations and analysis are available in the Jupyter notebook, which is still a work in progress, available at my GitHub page <a href="https://github.com/mjboothaus/blog.git" target="_blank" rel="noopener">https://github.com/mjboothaus/blog.git.</a>

### Personal notes

While I have always found the Titanic story to be both tragic and instructional, my uncle was an enthusiastic and long-time member of the official <a href="http://www.titanichistoricalsociety.org" target="_blank" rel="noopener">Titanic Historical Society</a> as the “Certificate of Appreciation” acknowledges. He had a large model replica of the Titanic in his house which always fascinated me as a child (and as an adult). Below are some pictures from his collection.

<img loading="lazy" class="alignnone size-full wp-image-235" src="https://i1.wp.com/mjboothaus.wpcomstaging.com/wp-content/uploads/2017/06/titanic2-e1498922251343.jpg?resize=730%2C402&#038;ssl=1" alt="Titanic2" width="730" height="402" data-recalc-dims="1" /> 

<img loading="lazy" class="alignnone size-full wp-image-236" src="https://i0.wp.com/mjboothaus.wpcomstaging.com/wp-content/uploads/2017/06/titanic4-e1498922331801.jpg?resize=730%2C969&#038;ssl=1" alt="Titanic Historical Society - Certificate of Appreciation" width="730" height="969" data-recalc-dims="1" />

 [1]: http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.csv