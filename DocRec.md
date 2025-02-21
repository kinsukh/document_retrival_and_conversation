DocRec 

it is a chatbot that enable you to search things in a pdf and get the result in a chatbot interface.
it is a simple chatbot that can be used to search for information in a pdf document. 

it stores the pdf in a database and then uses the pdfminer library to extract the text from the pdf.
and then uses the nltk library to tokenize the text and then uses the cosine similarity to find the most similar text to the query.
i am using the flask library to create a web interface for the chatbot.
and llm models to create the chatbot interface.
also i am using pine cone to store the pdfs in a database. and retrive vector emmbeddings for the pdfs.
and from this embedings i can find the most similar pdf part to the query. and then i can extract the text from the pdf and return it to the user.

the chatbot is a simple chatbot that can be used to search for information in a pdf document.