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
  return node.querySelector('.js-tweet-text-container');
}

function colorPredictions(messages = []) {
  const _messages = messages.map(({ text }) => text);
  // new Promise(resolve => setTimeout(resolve, 1000))
  fetch('http://localhost:9090/predict', {
    method: 'POST',
    body: JSON.stringify({ messages: _messages }),
    headers: {
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*'
    }
  })
    .then(response => response.json())
    .then(predictions => {
      messages.forEach((message, index) => {
        console.info(predictions[index]);

        colorTweet({ ...message, prediction: predictions[index] });
      });
    })
    .catch(err => {
      console.warn(err);
    });
}

function colorTweet(message) {
  document.querySelector(
    `li [data-item-id="${message.id}"]`
  ).style.backgroundColor = message.prediction ? 'red' : 'green';
}
