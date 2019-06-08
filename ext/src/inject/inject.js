chrome.extension.sendMessage({}, function(response) {
  var readyStateCheckInterval = setInterval(function() {
    if (document.readyState === 'complete') {
      clearInterval(readyStateCheckInterval);

      // ----------------------------------------------------------
      // This part of the script triggers when page is done loading
      console.log('Loaded Extension Inject');
      // ----------------------------------------------------------
      pollTrollOrNot();
    }
  }, 10);
});

function pollTrollOrNot() {
  setInterval(trollOrNot, 1000);
}

function trollOrNot() {
  const nodes = getPageMessageNodes();

  const messages = Array.from(nodes)
    .filter(withTextContainer)
    .map(toMessages);

  colorPredictions(messages);
}

function getPageMessageNodes() {
  return document.querySelectorAll('li [data-item-id]');
}

function toMessages(node) {
  const id = node.attributes['data-item-id'].value;
  const text = node.querySelector('.js-tweet-text-container').innerText;

  return { id, text };
}

function withTextContainer(node) {
  return !!node.querySelector('.js-tweet-text-container');
}

function extractText({ text }) {
  return text;
}

function getPredictions(messages) {
  return fetch('http://localhost:9090/predict', {
    method: 'POST',
    body: JSON.stringify({ messages: messages.map(extractText) }),
    headers: {
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*'
    }
  }).then(response => response.json());
}

function colorPredictions(messages = []) {
  getPredictions(messages)
    .then(predictions => {
      messages.forEach((message, index) => {
        colorTweet({ ...message, prediction: predictions[index] });
      });
    })
    .catch(onError);
}

function onError(err) {
  console.warn(err);
}

function colorTweet({ id, prediction }) {
  document.querySelector(`li [data-item-id="${id}"]`).style.backgroundColor = prediction
    ? 'red'
    : 'green';
}

module.exports = {
  pollTrollOrNot,
  trollOrNot,
  getPageMessageNodes,
  toMessages,
  withTextContainer,
  extractText,
  getPredictions,
  colorPredictions,
  onError,
  colorTweet
};
