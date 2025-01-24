// See https://github.com/dialogflow/dialogflow-fulfillment-nodejs
// for Dialogflow fulfillment library docs, samples, and to report issues
'use strict';
 
const functions = require('firebase-functions');
const {WebhookClient} = require('dialogflow-fulfillment');
const {Card, Suggestion} = require('dialogflow-fulfillment');
 
process.env.DEBUG = 'dialogflow:debug'; // enables lib debugging statements
 
exports.dialogflowFirebaseFulfillment = functions.https.onRequest((request, response) => {
  const agent = new WebhookClient({ request, response });
  console.log('Dialogflow Request headers: ' + JSON.stringify(request.headers));
  console.log('Dialogflow Request body: ' + JSON.stringify(request.body));
 
  function welcome(agent) {
    agent.add(`Welcome to my agent!`);
  }
 
  function fallback(agent) {
    agent.add(`I didn't understand`);
    agent.add(`I'm sorry, can you try again?`);
  }

 function gotomovie(agent) {
     agent.add(`This message is from Coffee Shop movie fans!`);
     agent.add(new Card({
     title: `A blog for Coffee Shops with movies to watch with friends and family`,
     imageUrl: 'https://www.eco-ai-horizons.com/coffeeshop.jpg',
     text: `The button takes you to the movie selection we have\n  Great to have you here! 💁`,
     buttonText: 'Click here',
     buttonUrl: 'https://www.primevideo.com/'
      })
     );
   }
  // Run the proper function handler based on the matched Dialogflow intent name
  let intentMap = new Map();
  intentMap.set('Default Welcome Intent', welcome);
  intentMap.set('Default Fallback Intent', fallback);
  intentMap.set('choose_movie-yes-yes', gotomovie);
  
  agent.handleRequest(intentMap);
});
