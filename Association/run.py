#!/usr/bin/env python
# coding: utf-8

# ## Q1

# In[ ]:


import pandas as pd
from mlxtend.frequent_patterns import apriori


# In[2]:


gpa_df = pd.read_csv(r".\specs\gpa_question1.csv")


# In[3]:


gpa_df.head()


# In[4]:


gpa_df = gpa_df.drop('count', axis=1)


# In[5]:


gpa_df


# In[6]:


cat_df = pd.get_dummies(gpa_df, prefix = 'category')


# In[7]:


cat_df


# In[8]:


freq_itemsets = apriori(cat_df, min_support=0.15 ,use_colnames=True)


# In[9]:


export = freq_itemsets.to_csv(r'.\output\question1_out_apriori.csv',index=False)


# In[10]:


from mlxtend.frequent_patterns import association_rules


# In[11]:


ass_rule_9 = association_rules(freq_itemsets, metric="confidence", min_threshold=0.9)
ass_rule_9


# In[12]:


export = ass_rule_9.to_csv(r'.\output\question1_out_rules9.csv',index=False)


# In[13]:


ass_rule_7 = association_rules(freq_itemsets, metric="confidence", min_threshold=0.7)
ass_rule_7


# In[14]:


export = ass_rule_7.to_csv(r'.\output\question1_out_rules7.csv',index=False)


# ## Q2 FP Growth

# In[43]:


import pandas as pd
file= pd.read_csv(r".\specs\bank_data_question2.csv")
file


# In[44]:


df= file.drop('id',axis=1)
df


# In[45]:


df['age'] = pd.cut(df['age'],3)
df['income'] = pd.cut(df['income'],3)
df['children'] = pd.cut(df['children'],3)


# In[46]:


df


# In[47]:


category_df = pd.get_dummies(df)


# In[48]:


category_df


# In[49]:


from mlxtend.frequent_patterns import fpgrowth
frequent_sets = fpgrowth(category_df, min_support=0.2,use_colnames=True)
frequent_sets


# In[50]:


from mlxtend.frequent_patterns import association_rules
ass_rule10= association_rules(frequent_sets, metric="confidence", min_threshold=0.7)
ass_rule10


# In[51]:


ass_rule10.to_csv(r'.\output\question2_out_rules.csv',index=False)


# In[ ]:




