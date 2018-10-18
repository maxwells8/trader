# Zeus
The FOREX trading AI platform.

*\*\*\* This project is unstable and will likely experience breaking changes*

#### What It Is Not
A batteries included trading AI.

#### What It Is
A set of tools to assist in the training, testing, and running of a FOREX trading AI.

#### Overview
Zeus is used by defining a "strategy". A strategy is made up of three parts: 
a Connector, a Trainer, and a Trader. A Connector describes where the time series data
is going to come from and how it will be loaded. A Trainer uses the data provided by the
Connector to train an AI model. The Trader is then used to both test (when using 
historical data) and run (when using current market data) the trained AI model.

#### Feature Road map
Below are features that are not necessarily being worked on, but desired for future
iterations:
* Additional ML model types (currently only a classification model utilizing OHLC 
features exist)
* Ability to specify a strategy via JSON
* An interactive shell environment
* A GUI to view stats and time series data and possibly assign labels manually.