1. Reddit 'Mental health' labeled Dataset - df_all.csv <br>
![image](https://github.com/Laney422/CS5246Project8/assets/74254801/3bd26620-24a3-456b-8c8a-5e18810f9f06)
2. Suicide dataset - df_suicide.csv <br>
![image](https://github.com/Laney422/CS5246Project8/assets/74254801/129bdbf0-55e1-469b-9313-9885ba48409a)
3. In the df_nowk_with_prob.zip, there are two datasets with masked keywords.
   - Two columns for text:'text_no_kw' or 'cleaned_no_kw'. The latter is cleaned (such as remove stopwords, punctuation,etc.
   - How: using [MASK] replaced all keywords.
   - keywords = {
    'adhd': ['adhd', 'vyvanse','medication', 'diagnosed'], 
    'almosthomeless': [ 'homeless', 'help', 'eviction', 'car'],
    'anxiety': ['anxiety', 'medication'],
    'assistance': ['assist', 'help',  'money'],
    'bipolar': ['bipolar',  'depression'],
    'depression': ['depression'],
    'eatingdisorders': [ 'foods', 'weight', 'disorder', 'eating lot', 'ate lot'],
    'get_motivated': ['energy', 'outcome','motivate'],
     #'normal_positive': ['good', 'best', 'happy', 'birthday', 'friend'],
    'ocd': [ 'ocd',  'obsessive'],
    'ptsd': [ 'therapist'],
    'selfharm': ['cut', 'self harm', 'blade','harm'],
    'stress': ['stress'],
    #'mask':['doubt', 'depress', 'trouble', 'pressure', 'sadness', 'grief'], # --> from paper, may not suit our          purpose
    'suicide':['die','death','survive','suicide','kill']
   }
   - Result: For multiple classifications, the performance drops by 5% in the F1 Score, indicating underfitting caused by information loss.
   - Result: For the suicide dataset, however, removing keywords does not affect too much performance.
