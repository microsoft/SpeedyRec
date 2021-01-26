# Dataset Format

File Name | Description
------------- | -------------
DocFeatures.tsv  | The information of news articles
traindata/ProtoBuf_i.tsv  | The click histories and impression logs of users
testdata/impressions/ProtoBuf_i.tsv  | Impression logs
testdata/daily_recall/daily_news_x.tsv  | Candidate set of news for recall in the day *x*
testdata/daily_recall/history_positive_x.tsv  | Clicked news of users in the day *x*

##Train Data
- *DocFeatures.tsv*
>news_id \t category \t subcategory \t title \t abstract \t body \n
- *ProtoBuf_i.tsv*
>user_id \t impression#1:pos#1;pos#2 & neg#1;neg#2 | impression#2:pos#1;pos#2 & neg#1;neg#2 \n

##Test Data
- *DocFeatures.tsv*
>news_id \t category \t subcategory \t title \t abstract \t body \n
- impressions (for testing AUC)
  - *ProtoBuf_i.tsv*
    >user_id \t history:news#1;news#2 \t pos#1;pos#2 \t neg#1;neg#2 \n
- daily_recall (for testing Recall)
  - *daily_news_x.tsv*
    >news#1 \t news#2 \t news#3
  - *history_positive_x.tsv*
    >user_id \t history:news#1;news#2 \t pos#1;pos#2 \n