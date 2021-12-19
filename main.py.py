
import pandas as pd
from warnings import filterwarnings
filterwarnings('ignore')


a=pd.read_excel("veriler.xlsx")


with pd.option_context("display.max_colwidth",None):

#noktalama işaretleri
a["yorumlar"] = a["yorumlar"].str.replace('[^\w\s]','')

#sayılar
a["yorumlar"]= a["yorumlar"].str.replace('\d','')

#stopwords
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
sw = stopwords.words('english')
a["yorumlar"]= a["yorumlar"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))

#lemmi
from textblob import Word
#nltk.download('wordnet')
a["yorumlar"] = a["yorumlar"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()])) 

#buyuk-kucuk donusumu
a["yorumlar"]= a["yorumlar"].apply(lambda x: " ".join(x.lower() for x in x.split()))



with pd.option_context("display.max_colwidth",None):
    display(a)


from textblob import TextBlob

a["sentiment"]=a["yorumlar_en"].map(lambda x :TextBlob(x).sentiment.polarity)


for i in a["sentiment"]:
    if i <0:
        a["sentiment"].replace(to_replace=i,value="Negatif",inplace=True)
    elif i>0:
        a["sentiment"].replace(to_replace=i,value="Pozitif",inplace=True)
    else:
        a["sentiment"].replace(to_replace=i,value="Nötr",inplace=True)



a.head(20)



a["sentiment"].value_counts()


# In[37]:


a["sentiment"].value_counts().plot(kind="bar")


# In[38]:


a.groupby(["markalar","sentiment"])[["sentiment"]].count()


# In[42]:


a.groupby(["markalar","sentiment"])[["sentiment"]].count().plot.barh()


# In[39]:


pd.crosstab(a["markalar"],a["sentiment"]).apply(lambda r:r/r.sum(),axis=1)

