# this function triggers an endpoint to a model trained on SageMaker with BlazingText

import boto3
import json
import re

def simple_tokenizer(input_text):
    REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(=)|(`)")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)|(\n)|(\t)")
    tokens = REPLACE_NO_SPACE.sub("", input_text.lower())
    tokens = REPLACE_WITH_SPACE.sub(" ", tokens) # note that blazing text expects space-separated tokens
    return tokens

# primary function
def lambda_handler(event, context):
    # lambda receives the input from the web app as an event in json format
    sentences = event['body']
    # sentences=list(sentences)
    tokenized_sentences = [simple_tokenizer(sentences)]
    payload = {"instances" : tokenized_sentences}
    
    # The SageMaker runtime is what allows us to invoke the endpoint that we've created.
    runtime = boto3.Session().client('sagemaker-runtime')

    # Now we use the SageMaker runtime to invoke our endpoint, sending the text we were given
    response = runtime.invoke_endpoint(EndpointName = 'blazingtext-2020-08-13-16-48-57-681',# name of the endpoint we created
                                      ContentType = 'application/json',                 # data format expected by BlazingText
                                      Body = json.dumps(payload))

    # The response is an HTTP response whose body contains the result of our inference
    output = json.loads(response['Body'].read().decode('utf-8'))    
    prob = output[0]['prob'][0]*100
    label = output[0]['label'][0].split('__label__')[1]
    
    
    if label=='0':
        response = runtime.invoke_endpoint(EndpointName = 'blazingtext-2020-08-13-18-57-15-339',
                                          ContentType = 'application/json',                 
                                          Body = json.dumps(payload))

        output = json.loads(response['Body'].read().decode('utf-8'))    
        subprob = output[0]['prob'][0]*100
        topic = output[0]['label'][0].split('__label__')[1]   
        output = 'Prediction: NEGATIVE sentiment with probability {:.1f}%, Topic {} with probability {:.1f}%'.format(prob, topic, subprob)
    elif label=='1':
        response = runtime.invoke_endpoint(EndpointName = 'blazingtext-2020-08-13-20-36-38-040',
                                          ContentType = 'application/json',                 
                                          Body = json.dumps(payload))

        output = json.loads(response['Body'].read().decode('utf-8'))    
        subprob = output[0]['prob'][0]*100
        topic = output[0]['label'][0].split('__label__')[1]   
        output = 'Prediction: POSITIVE sentiment with probability {:.1f}%, Topic {} with probability {:.1f}%'.format(prob, topic, subprob)
    else:
        output = 'Error occurred'
    
    # we return the output in a format expected by API Gateway
    return {
        'statusCode' : 200,
        'headers' : { 'Content-Type' : 'text/plain', 'Access-Control-Allow-Origin' : '*' },
        'body' : output
    }