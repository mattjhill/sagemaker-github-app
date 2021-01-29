import json
import torch
from torch import nn
import transformers
from transformers import OpenAIGPTModel, OpenAIGPTPreTrainedModel, OpenAIGPTConfig, Trainer, TrainingArguments
import os
import numpy as np
import pandas
import alpaca_trade_api

json.encoder.FLOAT_REPR = lambda o: format(o, '.2f')

api = alpaca_trade_api.REST()
class StockPredictionModelHighLow(OpenAIGPTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.linear = nn.Linear(5, config.n_embd)
        self.transformer = OpenAIGPTModel(config)
        self.output = nn.Linear(config.n_embd, 2)
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        charts=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None
    ):  
        
        inputs_embeds = self.linear(charts)
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        prediction = self.output(transformer_outputs.last_hidden_state)
        
        #  We are doing regression
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(labels, prediction)
            
            return loss, prediction
        else:
            return prediction
    
model = StockPredictionModelHighLow.from_pretrained('model_data/')

def lambda_handler(event, context):
    """Sample pure Lambda function

    Parameters
    ----------
    event: dict, required
        API Gateway Lambda Proxy Input Format

        Event doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html#api-gateway-simple-proxy-for-lambda-input-format

    context: object, required
        Lambda Context runtime methods and attributes

        Context doc: https://docs.aws.amazon.com/lambda/latest/dg/python-context-object.html

    Returns
    ------
    API Gateway Lambda Proxy Output Format: dict

        Return doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html
    """

    # chart_dict = json.loads(event['body'])
    ticker = event['pathParameters']['ticker']
    date = event['pathParameters']['date']
    chart_dict = api.polygon.historic_agg_v2(ticker, 5, 'minute', date,  date).df.between_time('9:30', '15:55').to_dict(orient='records')
    chart = pandas.DataFrame(chart_dict)
    # convert to PyTorch data types.
    chart = chart[['open', 'high', 'low', 'close', 'volume']].values
    chart = np.log(chart)
    log_open = chart[0, 0]
    chart[:, :-1] -= chart[:1, :1]
    chart[:, :-1] *= 100
    chart[:, -1] -= chart[:1, -1]
    chart = torch.tensor(chart, dtype=torch.float)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    chart = chart.to(device)
    with torch.no_grad():
        prediction_output = model(charts=chart)

    prediction_output /= 100
    prediction_output += log_open
    prediction_output = np.exp(prediction_output)
    for chart_item, prediction in zip(chart_dict, prediction_output.tolist()):
        chart_item['projected_high'] = round(prediction[0], 3)
        chart_item['projected_low'] = round(prediction[1], 3)

    return {
        "statusCode": 200,
        "body": json.dumps({
            "ticker": ticker,
            "results": chart_dict
        }),
    }

