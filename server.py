import json
import simple_http_server.server as server
from simple_http_server import JSONBody
from simple_http_server import request_map
from simple_http_server import Response
from trollaganda import setup
from parse_args import parseArgs

predictor = None

def setupPredictor():
    global predictor
    args = parseArgs()
    predictor = setup(args=args)



@request_map("/predict", method=["POST"])
def my_ctrl2(data = JSONBody()):
    predictions = predictor.predict(messages=data["messages"])
    results = list(map(lambda x: True if x[0] else False, predictions))

    resultsJSON = json.dumps(results)

    return Response(status_code=200, headers={"Access-Control-Allow-Origin": "*", "Content-Type": "application/json", "Access-Control-Allow-Headers": "*"}, body=resultsJSON)


@request_map("/predict", method=["OPTIONS"])
def my_ctrl1():
    return Response(status_code=200, headers={"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Headers": "*", "Content-Type": "application/json"})


def main(*args):
    setupPredictor()
    server.start()


if __name__ == "__main__":
    main()
