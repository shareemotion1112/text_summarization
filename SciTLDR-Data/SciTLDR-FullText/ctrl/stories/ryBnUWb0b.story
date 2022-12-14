In cities with tall buildings, emergency responders need an accurate floor level location to find 911 callers quickly.

We introduce a system to estimate a victim's floor level via their mobile device's sensor data in a two-step process.

First, we train a neural network to determine when a smartphone enters or exits a building via GPS signal changes.

Second, we use a barometer equipped smartphone to measure the change in barometric pressure from the entrance of the building to the victim's indoor location.

Unlike impractical previous approaches, our system is the first that does not require the use of beacons, prior knowledge of the building infrastructure, or knowledge of user behavior.

We demonstrate real-world feasibility through 63 experiments across five different tall buildings throughout New York City where our system predicted the correct floor level with 100% accuracy.

Indoor caller floor level location plays a critical role during 911 emergency calls.

In one use case, it can help pinpoint heart attack victims or a child calling on behalf of an incapacitated adult.

In another use case, it can help locate firefighters and other emergency personnel within a tall or burning building.

In cities with tall buildings, traditional methods that rely on GPS or Wi-Fi fail to provide reliable accuracy for these situations.

In these emergency situations knowing the floor level of a victim can speed up the victim search by a factor proportional to the number of floors in that building.

In recent years methods that rely on smartphone sensors and Wi-Fi positioning BID28 have been used to formulate solutions to this problem.

In this paper we introduce a system that delivers an estimated floor level by combining deep learning with barometric pressure data obtained from a Bosch bmp280 sensor designed for "floor level accuracy" BID3 and available in most smartphones today 1 .

We make two contributions: the first is an LSTM BID13 trained to classify a smartphone as either indoors or outdoors (IO) using GPS, RSSI, and magnetometer sensor readings.

Our model improves on a previous classifier developed by BID1 .

We compare the LSTM against feedforward neural networks, logistic regression, SVM, HMM and Random Forests as baselines.

The second is an algorithm that uses the classifier output to measure the change in barometric pressure of the smartphone from the building entrance to the smartphone's current location within the building.

The algorithm estimates the floor level by clustering height measurements through repeated building visits or a heuristic value detailed in section 4.5.We designed our method to provide the most accurate floor level estimate without relying on external sensors placed inside the building, prior knowledge of the building, or user movement behavior.

It merely relies on a smartphone equipped with GPS and barometer sensors and assumes an arbitrary user could use our system at a random time and place.

We offer an extensive discussion of potential real-world problems and provide solutions in (appendix B).We conducted 63 test experiments across six different buildings in New York City to show that the system can estimate the floor level inside a building with 65.0% accuracy when the floor-ceiling distance in the building is unknown.

However, when repeated visit data can be collected, our simple clustering method can learn the floor-ceiling distances and improve the accuracy to 100%.

All code, data and data collection app are available open-source on github.2 .

Current approaches used to identify floor level location fall into two broad categories.

The first method classifies user activity, i.e., walking, stairs, elevator, and generates a prediction based on movement offset BID1 .

The second category uses a barometer to calculate altitude or relative changes between multiple barometers BID20 BID17 BID27 .

We note that elevation can also be estimated using GPS.

Although GPS works well for horizontal coordinates (latitude and longitude), GPS is not accurate in urban settings with tall buildings and provides inaccurate vertical location BID16 ).

BID26 describe three modules which model the mode of ascent as either elevator, stairs or escalator.

Although these methods show early signs of success, they required infrastructure support and tailored tuning for each building.

For example, the iOS app BID26 used in this experiment requires that the user state the floor height and lobby height to generate predictions.

BID1 use a classifier to detect whether the user is indoors or outdoors.

Another classifier identifies whether a user is walking, climbing stairs or standing still.

For the elevator problem, they build another classifier and attempt to measure the displacement via thresholding.

While this method shows promise, it needs to be calibrated to the user's step size to achieve high accuracy, and the floor estimates rely on observing how long it takes a user to transition between floors.

This method also relies on pre-training on a specific user.

BID17 conduct a study of the feasibility of using barometric pressure to generate a prediction for floor level.

The author's first method measures the pressure difference between a reference barometer and a "roving" barometer.

The second method uses a single barometer as both the reference and rover barometer, and sets the initial pressure reading by detecting Wi-Fi points near the entrance.

This method also relies on knowing beforehand the names of the Wi-Fi access points near the building entrance.

BID27 equip a building with reference barometers on each floor.

Their method thus allows them to identify the floor level without knowing the floor height.

This technique also requires fitting the building with pressure sensors beforehand.

To our knowledge, there does not exist a dataset for predicting floor heights.

Thus, we built an iOS app named Sensory 3 to aggregate data from the smartphone's sensors.

We installed Sensory on an iPhone 6s and set to stream data at a rate of 1 sample per second.

Each datum consisted of the following: indoors, created at, session id, floor, RSSI strength, GPS latitude, GPS longitude, GPS vertical accuracy, GPS horizontal accuracy, GPS course, GPS speed, barometric relative altitude, barometric pressure, environment context, environment mean bldg floors, environment activity, city name, country name, magnet x, magnet y, magnet z, magnet total.

Each trial consisted of a continuous period of Sensory streaming.

We started and ended each trial by pressing the start button and stop button on the Sensory screen.

We separated data collection by two different motives: the first to train the classifier, the second to make floor level predictions.

The same sensory setup was used for both with two minor adjustments: 1) Locations used to train the classifier differed from locations used to make building predictions.

2) The indoor feature was only used to measure the classifier accuracy in the real world.

Our system operates on a time-series of sensor data collected by an iPhone 6s.

Although the iPhone has many sensors and measurements available, we only use six features as determined by forests of trees feature reduction BID14 .

Specifically, we monitor the smartphone's barometric pressure P , GPS vertical accuracy GV , GPS horizontal accuracy GH, GPS Speed GS, device RSSI 4 level rssi and magnetometer total reading M .

All these signals are gathered from the GPS transmitter, magnetometer and radio sensors embedded on the smartphone.

TAB4 shows an example of data points collected by our system.

We calculate the total magnetic field strength from the three-axis x, y, z provided by the magnetometer by using equation 1.

Appendix B.5 describes the data collection procedure.

DISPLAYFORM0 DISPLAYFORM1 The data used to predict the floor level was collected separately from the IO classifier data.

We treat the floor level dataset as the testing set used only to measure system performance.

We gathered 63 trials among five different buildings throughout New York City to explore the generality of our system.

Our building sampling strategy attempts to diversify the locations of buildings, building sizes and building material.

The buildings tested explore a wide-range of possible results because of the range of building heights found in New York City (Appendix 4).

As such, our experiments are a good representation of expected performance in a real-world deployment.

The procedure described in appendix B.6 generates data used to predict a floor change from the entrance floor to the end floor.

We count floor levels by setting the floor we enter to 1.

This trial can also be performed by starting indoors and walking outdoors.

In this case, our system would predict the person to be outdoors.

If a building has multiple entrances at different floor levels, our system may not give the correct numerical floor value as one would see in an elevator.

Our system will also be off in buildings that skip the 13th floor or have odd floor numbering.

The GPS lag tended to be less significant when going from inside to outside which made outside predictions trivial for our system.

As such, we focus our trials on the much harder outside-to-inside prediction problem.

To explore the feasibility and accuracy of our proposed clustering system we conducted 41 separate trials in the Uris Hall building using the same device across two different days.

We picked the floors to visit through a random number generator.

The only data we collected was the raw sensor data and did not tag the floor level.

We wanted to estimate the floor level via entirely unsupervised data to simulate a real-world deployment of the clustering mechanism.

We used both the elevator and stairs arbitrarily to move from the ground floor to the destination floor.

In this section, we present the overall system for estimating the floor level location inside a building using only readings from the smartphone sensors First, a classifier network classifies a device as either indoors or outdoors.

The next parts of the algorithm identify indoor/outdoor transitions (IO), measure relative vertical displacement based on the device's barometric pressure, and estimate absolute floor level via clustering.

From our overall signal sequence {x 1 , x 2 , x j , ..., x n } we classify a set of d consecutive sensor readings X i = {x 1 , x 2 , ..., x d } as y = 1 if the device is indoors or y = 0 if outdoors.

In our experiments we use the middle value x j of each X i as the y label such that DISPLAYFORM0 The idea is that the network learns the relationship for the given point by looking into the past and future around that point.

This design means our system has a lag of d/2 ??? 1 second before the network can give a valid prediction.

We chose d = 3 as the number of points in X by random-search BID2 over the point range [1, 30] .

Fixing the window to a small size d allows us to use other classifiers as baselines to the LSTM and helps the model perform well even over sensor reading sequences that can span multiple days.

The first key component of our system is a classifier which labels each device reading as either indoors or outdoors (IO).

This critical step allows us to focus on a subset of the user's activity, namely the part when they are indoors.

We conduct our floor predictions in this subspace only.

When a user is outside, we assume they are at ground level.

Hence our system does not detect floor level in open spaces and may show a user who is on the roof as being at ground level.

We treat these scenarios as edge-cases and focus our attention on floor level location when indoors.

Although our baselines performed well, the neural networks outperformed on the test set.

Furthermore, the LSTM serves as the basis for future work to model the full system within the LSTM; therefore, we use a 3-layer LSTM as our primary classifier.

We train the LSTM to minimize the binary cross-entropy between the true indoor state y of example i and the LSTM predicted indoor state LSTM(X) =?? of example i.

This objective cost function C can be formulated as: DISPLAYFORM0 Figure 4.2 shows the overall architecture.

The final output of the LSTM is a time-series T = {t 1 , t 2 , ..., t i , t n } where each t i = 0, t i = 1 if the point is outside or inside respectively.

The IO classifier is the most critical part of our system.

The accuracy of the floor predictions depends on the IO transition prediction accuracy.

The classifier exploits the GPS signal, which does not cut out immediately when walking into a building.

We call this the "lag effect."

The lag effect hurts our system's effectiveness in 1-2 story houses, glass buildings, or ascend methods that may take the user near a glass wall.

A substantial lag means the user could have already moved between floors by the time the classifier catches up.

This will throw off results by 1-2 floor levels depending on how fast the device has moved between floors.

The same effect is problematic when the classifier prematurely transitions indoors as the device could have been walking up or down a sloped surface before entering the building.

We correct for most of these errors by looking at a window of size w around the exact classified transition point.

We use the minimum device barometric pressure in this window as our p 0 .

We set w = 20 based on our observations of lag between the real transition and our predicted transition during experiments.

This location fix delay was also observed by BID19 to be between 0 and 60 seconds with most GPS fixes happening between 0 and 20 seconds.

Figure 2: To find the indoor/outdoor transitions, we convolve filters V 1 , V 2 across timeseries of Indoor/Outdoor predictions T and pick each subset s i with a Jaccard distance ??? 0.4.

The transition t i is the middle index in set s i .Given our LSTM IO predictions, we now need to identify when a transition into or out of a building occurred.

This part of the system classifies a sub-sequence s i = T i:i+|V1| of LSTM predictions as either an IO transition or not.

Our classifier consists of two binary vector masks DISPLAYFORM0 that are convolved across T to identify each subset s i ??? S at which an IO transition occurred.

Each subset s i is a binary vector of in/out predictions.

We use the Jaccard similarity BID4 as an appropriate measure of distance between V 1 , V 2 and any s i .As we traverse each subset s i we add the beginning index b i of s i to B when the Jaccard distances J 1 ??? 0.4 or J 2 ??? 0.4 as given by Equation 5.We define J j , j = {1, 2} by DISPLAYFORM1 The Jaccard distances J 1 , J 2 were chosen through a grid search from [0.2, 0.8].

The length of the masks V 1 , V 2 were chosen through a grid search across the training data to minimize the number of false positives and maximize accuracy.

Once we have each beginning index b i of the range s i , we merge nearby b i s and use the middle of each set b as an index of T describing an indoor/outdoor transition.

At the end of this step, B contains all the IO transitions b into and out of the building.

The overall process is illustrated by FIG1 and described in Algorithm 1.

This part of the system determines the vertical offset measured in meters between the device's inbuilding location, and the device's last known IO transition.

In previous work suggested the use of a reference barometer or beacons as a way to determine the entrances to a building.

Our second key contribution is to use the LSTM IO predictions to help our system identify these indoor transitions into the building.

The LSTM provides a self-contained estimator of a building's entrance without relying on external sensor information on a user's body or beacons placed inside a building's lobby.

This algorithm starts by identifying the last known transition into a building.

This is relatively straightforward given the set of IO transitions B produced by the previous step in the system (section 4.3).

We can simply grab the last observation b n ??? B and set the reference pressure p 0 to the lowest device pressure value within a 15-second window around b n .

A 15-second window accounts for the observed 15-second lag that the GPS sensor needs to release the location lock from serving satellites.

The second datapoint we use in our estimate is the device's current pressure reading p 1 .

To generate the relative change in height m ??? we can use the international pressure equation FORMULA6 BID18 .

DISPLAYFORM0 As a final output of this step in our system we have a scalar value m ??? which represents the relative height displacement measured in meters, between the entrance of the building and the device's current location.

This final step converts the m ??? measurement from the previous step into an absolute floor level.

This specific problem is ill-defined because of the variability in building numbering systems.

Certain buildings may start counting floors at 1 while others at 0.

Some buildings also avoid labeling the 13th floor or a maintenance floor.

Heights between floors such as lobbies or food areas may be larger than the majority of the floors in the building.

It is therefore tough to derive an absolute floor number consistently without prior knowledge of the building infrastructure.

Instead, we predict a floor level indexed by the cluster number discovered by our system.

We expand on an idea explored by BID27 to generate a very accurate representation of floor heights between building floors through repeated building visits.

The authors used clusters of barometric pressure measurements to account for drift between sensors.

We generalize this concept to estimate the floor level of a device accurately.

First, we define the distance between two floors within a building d i,j as the tape-measure distance from carpet to carpet between floor i and floor j.

Our first solution aggregates m ??? estimates across various users and their visits to the building.

As the number M of m ??? 's increases, we approximate the true latent distribution of floor heights which we can estimate via the observed m ??? measurement clusters K. We generate each cluster k i ??? K by sorting all observed m ??? 's and grouping points that are within 1.5 meters of each other.

We pick 1.5 because it is a value which was too low to be an actual d i,j distance as observed from an 1107 building dataset of New York City buildings from the Council on tall buildings and urban habitat (sky, 2017).

During prediction time, we find the closest cluster k to the device's m ??? value and use the index of that cluster as the floor level.

Although this actual number may not match the labeled number in the building, it provides the true architectural floor level which may be off by one depending on the counting system used in that building.

Our results are surprisingly accurate and are described in-depth in section 5.When data from other users is unavailable, we simply divide the m ??? value by an estimatorm from the sky (2017) dataset.

Across the 1107 buildings, we found a bi-modal distribution corresponding to office and residential buildings.

For residential buildings we letm r = 3.24 andm o = 4.02 for office buildings, FIG3 shows the dataset distribution by building type.

If we don't know the type of building, we usem = 3.63 which is the mean of both estimates.

We give a summary of the overall algorithm in the appendix (2).

We separate our evaluation into two different tasks: The indoor-outdoor classification task and the floor level prediction task.

In the indoor-outdoor detection task we compare six different models, LSTM, feedforward neural networks, logistic regression, SVM, HMM and Random Forests.

In the floor level prediction task, we evaluate the full system.

In this first task, our goal is to predict whether a device is indoors or outdoors using data from the smartphone sensors.

All indoor-outdoor classifiers are trained and validated on data from 35 different trials for a total of 5082 data points.

The data collection process is described in section 3.1.

We used 80% training, 20% validation split.

We don't test with this data but instead test from separately collected data obtained from the procedure in section 3.1.1.We train the LSTM for 24 epochs with a batch size of 128 and optimize using Adam BID15 ) with a learning rate of 0.006.

We chose the learning-rate, number of layers, d size, number of hidden units and dropout through random search BID2 .

We designed the LSTM network architecture through experimentation and picked the best architecture based on validation performance.

Training logs were collected through the python test-tube library BID10 and are available in the GitHub repository.

LSTM architecture: Layers one and two have 50 neurons followed by a dropout layer set to 0.2.

Layer 3 has two neurons fed directly into a one-neuron feedforward layer with a sigmoid activation function.

TAB0 gives the performance for each classifier we tested.

The LSTM and feedforward models outperform all other baselines in the test set.

We measure our performance in terms of the number of floors traveled.

For each trial, the error between the target floor f and the predicted floorf is their absolute difference.

Our system does not report the absolute floor number as it may be different depending on where the user entered the building or how the floors are numbered within a building.

We ran two tests with different m values.

In the first experiment, we used m = m r = 4.02 across all buildings.

This heuristic predicted the correct floor level with 65% accuracy.

In the second experiment, we used a specific m value for each individual building.

This second experiment predicted the correct floor with 100% accuracy.

These results show that a proper m value can increase the accuracy dramatically.

Table 2 describes our results.

In each trial, we either walked up or down the stairs or took the elevator to the destination floor, according to the procedure outlined in section 3.1.1.

The system had no prior information about the buildings in these trials and made predictions solely from the classifier and barometric pressure difference.

1-2 5.17 5.46 2-3 3.5 3.66 3-4 3.4 3.66 4-5 3.45 3.5 5-6 3.38 3.5 6-7 3.5 3.5 7-8 3.47 3.5In this section, we show results for estimating the floor level through our clustering system.

The data collected here is described in detail in section 3.1.2.

In this particular building, the first floor is 5 meters away from the ground, while the next two subsequent floors have a distance of 3.65 meters and remainder floors a distance of 3.5.

To verify our estimates, we used a tape measure in the stairwell to measure the distance between floors from "carpet to carpet." TAB2 compares our estimates against the true measurements.

Figure 5 in the appendix shows the resulting k clusters from the trials in this experiment.

Separating the IO classification task from the floor prediction class allows the first part of our system to be adopted across different location problems.

Our future work will focus on modeling the complete problem within the LSTM to generate floor level predictions from raw sensor readings as inspired by the works of BID11 and BID12 .

In this paper we presented a system that predicted a device's floor level with 100% accuracy in 63 trials across New York City.

Unlike previous systems explored by BID1 , BID26 , BID20 , BID17 BID27 , our system is completely selfcontained and can generalize to various types of tall buildings which can exceed 19 or more stories.

This makes our system realistic for real-world deployment with no infrastructure support required.

We also introduced an LSTM, that solves the indoor-outdoor classification problem with 90.3% accuracy.

The LSTM matched our baseline feedforward network, and outperformed SVMs, random forests, logistic regression and previous systems designed by BID22 and BID30 .

The LSTM model also serves as a first step for future work modeling the overall system end-to-end within the LSTM.Finally, we showed that we could learn the distribution of floor heights within a building by aggregating m ??? measurements across different visits to the building.

This method allows us to generate precise floor level estimates via unsupervised methods.

Our overall system marries these various elements to make it a feasible approach to speed up real-world emergency rescues in cities with tall buildings.

In this section we explore potential pitfalls of our system in a real-world scenario and offer potential solutions.

Figure 3: Adjusting device pressure from readings from a nearby station.

The readings were mostly adjusted but the lack of resolution from the reference station made the estimate noisy throughout the experiment.

One of the main criticisms for barometric pressure based systems is the unpredictability of barometric pressure as a sensor measurement due to external factors and changing weather conditions.

Critics have cited the discrepancy between pressure-sealed buildings and their environments, weather pattern changes, and changes in pressure due to fires BID23 .

BID17 used a reference weather station at a nearby airport to correct weather-induced pressure drift.

They showed the ability to correct weather drift changes with a maximum error of 2.8 meters.

BID27 also used a similar approach but instead adjust their estimates by reading temperature measurements obtained from a local weather station.

We experimented with the method described by BID17 and conducted a single trial as a proof of concept.

We measured the pressure reading p from an iPhone device on a table over 7 hours while simultaneously querying weather data w every minute.

By applying the offset equation 7 we attempt to normalize the p i reading to the first p 0 reading generated by the device DISPLAYFORM0 we were able to stay close to the initial p 0 estimate over the experiment period.

We did find that the resolution from the local weather station needed to be fine-grained enough to keep the error from drifting excessively.

Figure 3 shows the result of our experiment.

B.2 TIME SENSITIVITY Our method works best when an offset m ??? is predicted within a short window of making a transition within the building.

BID17 explored the stability of pressure in the short term and found the pressure changed less than 0.1 hPa every 10 minutes BID17 on average.

The primary challenge arises in the case when a user does not leave their home for an extended period of hours.

In this situation, we can use the previously discussed weather offset method from section B.1, or via indoor navigation technology.

We can use the device to list Wi-Fi access points within the building and tag each cluster location using RSSI fingerprinting techniques as described by BID29 BID8 and BID25 .

With these tags in place, we can use the average floor level tags of the nearest n Wi-Fi access points once the delay between the building entrance and the last user location is substantial.

We could not test this theory directly because of the limitations Apple places on their API to show nearby Wi-Fi access points to non-approved developers.

Another potential source of error is the difference between barometric pressure device models.

BID27 conducted a thorough comparison between seven different barometer models.

They concluded that although there was significant variation between barometer accuracies, the errors were consistent and highly correlated with each device.

They also specifically mentioned that the Bosch BMP180 barometer, the older generation model to the one used in our research, provided the most accurate measurements from the other barometers tested.

In addition, BID17 also conducted a thorough analysis using four different barometers.

Their results are in line with BID27 , and show a high correlation and a constant offset between models.

They also noted that within the same model (Bosch BMP180) there was also a measurement variation but it was constant BID17 .

Our system relies on continuous GPS and motion data collected on the mobile device.

Continuously running the GPS and motion sensor on the background can have an adverse effect on battery life.

BID30 showed that GPS drained the battery roughly double as fast across three different devices.

Although GPS and battery technology has improved dramatically since 2012, GPS still has a large adverse effect on battery life.

This effect can vary across devices and software implementation.

For instance, on iOS, the system has a dedicated chip that continuously reads device sensor data BID9 .

This approach allows the system to stream motion events continuously without rapidly draining battery life.

GPS data, however, does not have the same hardware optimization and is known to drain battery life rapidly.

BID19 conducted a thorough study of the impact of running GPS continuously on a mobile device.

They propose a method based on adjusted sampling rates to decrease the negative impact of GPS on battery life.

For real-world deployment, this approach would still yield fairly fine-grained resolution and would have to be tuned by a device manufacturer for their specific device model.

<|TLDR|>

@highlight

We used an LSTM to detect when a smartphone walks into a building. Then we predict the device's floor level using data from sensors aboard the smartphone.

@highlight

The paper introduces a system to estimate a floor-level via their mobile device's sensor data using an LSTM and changes in barometric pressure

@highlight

Proposal for a two-step method to determine which floor a mobile phone is on inside a tall building.