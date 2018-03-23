/*

	Programmer:	Adam Mischke
	Professor:	Joshua L. Phillips
	Program:	Unsupervised Learning - Project 4
	Started:	Sunday,	Nov. 19 at 19:44:45
	Date Due:	Tuesday, Nov. 14 by 10:00pm
	Date Finished: Monday, Nov. 13 at 17:30:00

	K means Algorithm:
	1. Given a set of N training examples each consisting of a vector of continuous attributes, 
	   select K training examples to be a set of initial cluster means (centers)

	2. Determine the distance (many options to choose from here) between each training example
	   and each of the K cluster means
	
	3. Assign each training example to the closest cluster mean
	
	4. Calculate the average of the training examples assigned to each cluster mean (creates a new mean)
	
	5. Go back to step 2 until the cluster means do not change 
	   (i.e. all training examples are assigned to the same cluster mean as on the previous iteration) 


*/

#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <fstream>
#include <random>
#include <limits>
#include <sstream>
#include <iterator>
#include <set>

using std::vector;
using std::cout;
using std::string;
using std::cin;
using std::endl;
using std::ostream;

/* typedeffing for ease of use */
typedef vector<double> vd;
typedef vector<vd> vvd;
typedef vector<int> vi;
typedef vector<vector<int > > vvi;

// handles fstream from the filename into the 2d vector of doubles
void handleFile(const string fn, vvd &v);

// shuffles the traintable and initializes the centroid table with how ever many cluster we have
void initKmeans(vvd &centroidTable, vvd trainTable);

// returns true if centroids have not moved
// calculates the distances between the clusters and the points
bool calculateDistances(const vvd &train, vvd &centroid, vi& clusterTags);

// returns the euclidean distance of a particular vector
double calculateDistance(const vd &train, const vd &center);

// returns true if centroids have not moved
// moves the clusters means based on the number of rows in a cluster (avg)
bool updateClusters(const vvd &trainTable, vvd &centroidTable, const vi &clusterTags);

// goes through the training table and majority votes on which cluster goes with which class label
vi findLabels(const vvd &trainTable,const vi &clusterTags);

// returns how many are right
// tests our assigned labels of test with their actual from the testtable
int testLabels(const vvd &trainTable, const vvd &testTable, const vi &clusterTags, const vi &actualTable);

// returns which row it's in
// finds the closest point from the test table to the train table
int findClosestPoint(const vvd &trainTable, const vvd &testTable);

// comparison of two vectors
bool equal(const vvd &oldCentroids, const vvd &newCentroids);

/* PRINTING */
template <typename t> void print(vector<vector< t > >  v);
template <typename t> void print(vector< t >   v);

int SEED, NUM_CLUSTERS, NUM_FEATURES, NUM_CLASSIFICATIONS;

int main(int argc, char* argv[]){


	int numRight=0;

	// integer: random seed, 
	SEED = atoi(argv[1]);
	// SEED = 0;

	// integer: the number of clusters, 
	NUM_CLUSTERS = atoi(argv[2]);
	//NUM_CLUSTERS = 140;

	// integer: the number of real-valued features in the data set, 
	NUM_FEATURES = atoi(argv[3]);
	//NUM_FEATURES = 4;

	// string: input training data filename, 
	string trainFilename = argv[4];

	// string: input testing data filename
	string testFilename = argv[5];

	//cout << "initializing..\n";
	//cout << "seed: " << SEED << " num of clusters: " << NUM_CLUSTERS << " num of features: " << NUM_FEATURES << "\n";
	//cout << "training filename: " << trainFilename << " testing filename: " << testFilename << "\n";
	

	vvd trainTable;		// 2d vector of doubles
	vvd testTable;		// same
	vvd centroidTable;	// same

	vi clusterTags;		// 1d vector of clusters
	vi actualTable;		// actual classification of test table

	// read in the training data into the training table
	handleFile(trainFilename, trainTable);

   	std::set<int> result;
   	for (int i=0; i<trainTable.size(); i++){
   		result.insert(trainTable[i][NUM_FEATURES]);
   	}
    NUM_CLASSIFICATIONS = result.size();
	//cout << NUM_CLASSIFICATIONS << "\n";

	// same with the testing data
	handleFile(testFilename, testTable);

	// inaitialize the centroid table with random doubles
	initKmeans(centroidTable, trainTable);

	int moves=0;	// iterator
	
	// we need the last cluster tags to help finding and classifying labels 
	while ( !calculateDistances(trainTable, centroidTable, clusterTags) )
	{
		moves++;
	}
	

	//cout << "num of moves: " << moves << "\n";

	// assign the cluster means to an actual class label
	actualTable = findLabels(trainTable, clusterTags);

	// returns how many are right
	numRight = testLabels(trainTable, testTable, clusterTags, actualTable);

	cout << numRight << endl;
	
}

// returns the position of the point (row in traintable) of which the closest point of the test point is 
int findClosestPoint(const vvd &trainTable, const vd &test){

	double dist=0.0, closestPoint=0.0;
	int indice=0;

	// set the closest point to inf
	closestPoint = std::numeric_limits<double>::max(); 

	for (int j=0; j<trainTable.size(); j++){
		dist = calculateDistance(trainTable[j], test);		// find the distance between the two sets

		// if the distance we just got is the lowest in the clusters,
		if (dist < closestPoint)
		{
			closestPoint = dist;			// set some values
			indice = j;
		}
	}
	//cout << "dist: " << closestPoint << "\n";
	return indice;		// return the indice of the closest point
}

// tests the labels from the test table
int testLabels(const vvd &trainTable, const vvd &testTable, const vi &clusterTags, const vi &actualTable){

	int classification=0,guess=0,modGuess=0,correct=0;
	vi guessList;
	//cout << "oldClassification:\n";
	for (int i=0; i<testTable.size(); i++){

		// returns the indice of the closest point from the test table set
		// from the entirety of the training table
		classification = findClosestPoint(trainTable, testTable[i]);

		// pick the guess from the position found
		guess = clusterTags[classification];

		// modify the guess from the actual table
		modGuess = actualTable[guess]; 

		// push the modified guess to the guess list
		guessList.push_back(modGuess);
	}
	//cout << "\n";
	//print(guessList);

	// iterate between both sets and count the correct guesses
	for (int i=0; i<testTable.size(); i++){
		//cout << "guess: " << guessList[i] << " actual: " << testTable[i][NUM_FEATURES] << "\n";
		if(guessList[i] == testTable[i][NUM_FEATURES])
			correct++;	// increase the correct counter if the classification is correct
	}
	return correct;	// return how many are correct
}

// finds the labels of the training table and returns the actual labels associated with the classificatons
vi findLabels(const vvd &trainTable, const vi &clusterTags){

	vvi countTable(NUM_CLUSTERS, vector<int> (NUM_CLASSIFICATIONS));	// temp 2d vector used to count the probability of which classification 
																// belongs to a given cluster mean
	vi actualTable(NUM_CLUSTERS);								// table used to return the actual classification

	int actual = 0, tag = 0; // actual

	// iterate throught the training table..
	for (int r=0; r<trainTable.size(); r++){
		actual = trainTable[r][NUM_FEATURES];		// the actual classification for that training set
		tag = clusterTags[r];						// the tag associated with the training set
		countTable[tag][actual]++;					// increase the count withe the actual as the row and the tag as the column
	}

	// iterate through the number of cluster means
	for (int r=0; r<NUM_CLUSTERS; r++){
		// class label will be assigned to each vector by taking a majority vote amongst it's assigned examples 
		// from the training set (ties will be broken by preferring the smallest integer class label).
		actualTable[r] = std::distance(countTable[r].begin(), std::max_element(countTable[r].begin(), countTable[r].end()));
	}

	return actualTable;	// return the key map for the actual classifications
}

// returns whether centroids from old are the same as the new
// if it is, we'll break from the main loop
bool equal(const vvd &oldCentroids, const vvd &newCentroids){

	int rows = newCentroids.size();			// # of rows
	int columns = newCentroids[0].size();	// # of columns

	for(int i=0; i<rows; i++){
		for(int j=0; j<columns; j++){
			//if ( fabs( newCentroids[i][j] - oldCentroids[i][j] ) < std::numeric_limits<double>::min() )		// is this close enough?
				// return false;
			//cout << "new " << newCentroids[i][j] << " old " << oldCentroids[i][j] << "\n";
			if (newCentroids[i][j] != oldCentroids[i][j])	// c++11 allows comparison operator
				return false;		
		}
	}
	return true;
}

// updates the cluster means by averaging all of the rows that fit the cluster tags from the centroid table
bool updateClusters(const vvd &trainTable, vvd &centroidTable, const vi &clusterTags){

	vi countedClusters;		// initialize a new vector of ints for the number of clusters in each classification
	bool done = false;		// flag for when to exit the loop in main

	// create a new table for the centroids
	vvd newCentroids(NUM_CLUSTERS, vector<double> (NUM_FEATURES));


	// iterate through the rows
	for(int i=0; i<trainTable.size(); i++){
		// and columns of the centroid table
		for(int j=0; j<centroidTable[0].size(); j++){
			// increment the new centroid table based on the training table value
			// use the clustertag as the row and the iterator as the column
			newCentroids[ clusterTags[i] ][j] += trainTable[i][j];
		}
	}

	for (int i=0; i<NUM_CLUSTERS; i++){
		countedClusters.push_back( count(clusterTags.begin(), clusterTags.end(), i) );
	}

	//print (countedClusters);
	// divides each value by it's average
	for(int i=0; i<newCentroids.size(); i++){
		for(int j=0; j<newCentroids[0].size(); j++){

			if (countedClusters[i] != 0)		// prevent nan if the number near the cluster is 0
			{
				newCentroids[i][j] /= countedClusters[i];
			}
			else
			{
				newCentroids[i][j] = centroidTable[i][j];	// use the old one
			}


		}
	}

	// if the number of the old and new cluster are the same, the centroids have not moved
	if (equal(centroidTable, newCentroids) ){
		done = true;
		//cout << "done!\n";		// therefore, set that we are done and return
		return done;
	}

	centroidTable = newCentroids;	// updates the centroid table

	return done;					// returns false if we get this far
}

// distance function between two vectors
double calculateDistance(const vd &train, const vd &center){
	double total = 0.0;	// initialize a distance

	// iterate through each column
	for (int i=0; i<NUM_FEATURES; i++)
		total+= pow(center[i] - train[i], 2);	// take the two components, subtract and raise to the second power

	return sqrt(total);	// return the square root of the total
}

// calulates the distances from each cluster mean and their respective points from the training table
bool calculateDistances(const vvd &trainTable, vvd &centroidTable, vi &clusterTags){

	vd centroid;				// one center information
	vd training;				// one training information
	double dist = 0.0;			// distace calculated between a row in the center and a row in the train table
	double lowestDist = 0.0;	// lowest one in the whole training table
	int lowestCentroid = 0;		// the lowest row number in the centroid table
	int lowestRow = 0;			// the lowest row number in the train table
	bool done = false;			// flag for when to exit the while loop in main
	clusterTags.clear();		// reset the cluster tags

	// for each row in the training table
	for (int r=0; r<trainTable.size(); r++){

		lowestDist = std::numeric_limits<double>::max(); 	// reset distance
		training = trainTable[r];							// grab the row

		// for each cluster..
		for (int c=0; c<NUM_CLUSTERS; c++){
			centroid = centroidTable[c];					// grab the other row

			// pass the rows to a function that calculates the distance 
			dist = calculateDistance(training, centroid);
			//cout << "distance from: [" << r << ", " << c << "] is: " << dist << "\n";

			// if the distance we just got is the lowest in the clusters,
			if (dist < lowestDist)
			{
				lowestDist = dist;			// set some values
				lowestCentroid = c;
				lowestRow = r;
			}
		}
		//cout << "winner is..[" << lowestRow << ", " << lowestCentroid << "] is: " << lowestDist << "\n";
		clusterTags.push_back(lowestCentroid);	// then push the lowest center into a vector of tags

	}
	// update the means of the clusters
	done = updateClusters(trainTable, centroidTable, clusterTags);

	return done;	// return whether the clusters haven't been updated
}

// initializes the cluster means by picking a random row from the training table
void initKmeans(vvd &centroidTable, vvd trainTable){

  	std::mt19937 gen(SEED);  	 // set a good seed
	
	//  good init, but shuffle the train table..
	std::shuffle(std::begin(trainTable), std::end(trainTable), gen);
	for (int i=0; i<NUM_CLUSTERS; i++){
		vd tempVec;
		for (int j=0; j<NUM_FEATURES; j++){
			tempVec.push_back(trainTable[i][j]);
		} 
		centroidTable.push_back(tempVec);
	}
	
	//cout << "\n";
}

// handles the file input by opening an input filestream to open a string filename
// then puts the values of doubles into a 2 dimensional vector
void handleFile(const string fn, vvd &v){
	std::ifstream in;		// input filestream
	in.open(fn);			// open

	for (string line; std::getline(in, line); )
	{
    std::istringstream iss(line);
    v.emplace_back(std::istream_iterator<double>(iss),	// feed into the temp double from above
                   std::istream_iterator<double>());
	}
	in.close();				// close
}

/* Templated Print Statements for double and single vectors */
template <typename t>
void print(vector<vector< t > >  v){
	for (auto& i : v){
		for (auto& j : i){	
			cout << j << " ";
		}
		cout << "\n";
	}cout << "\n";

}
template <typename t>
void print(vector< t >   v){
	for (auto& i : v){
		cout << i << " ";
	}
	cout << "\n";
}
