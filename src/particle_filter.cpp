/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <map>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates
    //  of x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    
    // set number of particles
    num_particles = 100;
    
    //random distribution generator
    default_random_engine gen;
    
    //create normal distribution for x, y, and theta
    normal_distribution<double> dist_x(0, std[0]);
    normal_distribution<double> dist_y(0, std[1]);
    normal_distribution<double> dist_theta(0, std[2]);
    
    //define each particle with a normal distribution from initial GPS measurement
    for(int i=0;i<num_particles;i++){
        Particle p;
        p.id = i;
        p.x = x + dist_x(gen);
        p.y = y + dist_y(gen);
        p.theta = theta + dist_theta(gen);
        p.weight = 1.0;
        particles.push_back(p); //fill the particles vector with generated p
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/
    
    //random distribution generator
    default_random_engine gen;
    
    //create normal distribution for x, y, and theta
    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);
    
    //update pose of particles based on motion predition
    for(int i=0;i<num_particles;i++){
        // checking for zero yaw_rate
        if (fabs(yaw_rate) < 0.00001) {
            particles[i].x += velocity * delta_t * cos(particles[i].theta);
            particles[i].y += velocity * delta_t * sin(particles[i].theta);
        }
        else{
            particles[i].x += velocity/yaw_rate*(sin(particles[i].theta + yaw_rate*delta_t)
                                                 - sin(particles[i].theta));
            particles[i].y += velocity/yaw_rate*(cos(particles[i].theta)
                                                 - cos(particles[i].theta + yaw_rate*delta_t));
            particles[i].theta += yaw_rate*delta_t;
        }
    }
    
    //add normal distributed noise to particle movement
    for(int i=0;i<num_particles;i++){
        particles[i].x += dist_x(gen);
        particles[i].y += dist_y(gen);
        particles[i].theta += dist_theta(gen);
    }
    
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.
    
    double distance; //distance between predicted and observation
    double nearest_dist; //store nearest distance
    int nearest_index; //store index of nearest neighbor
    double x_dist; //x distance
    double y_dist; //y distance
    
    //get the nearest predicted neighbor of each observation
    for(int i=0;i<observations.size();i++){
        //initialize nearest distance for first point
        //nearest_dist = sqrt((observations[i].x - predicted[0].x)*(observations[i].x - predicted[0].x) + (observations[i].y - predicted[0].y)*(observations[i].y - predicted[0].y));
        //initialize nearest distance to a large number
        nearest_dist = 10000000.0;
        
        for(int j=0;j<predicted.size();j++){
            x_dist = observations[i].x - predicted[j].x;
            y_dist = observations[i].y - predicted[j].y;
            distance = sqrt(x_dist*x_dist + y_dist*y_dist);
            if(distance < nearest_dist){
                nearest_dist = distance;
                nearest_index = j;
            }
            //cout << "nearest distance: " << nearest_dist << endl;
        }
        observations[i].id = predicted[nearest_index].id;
        //cout << "observation id: " << observations[i].id << "x: " << observations[i].x << "y: " << observations[i].y << endl;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html
    
    //define variables, x and y of observation, x,y and theta of particles
    double x_obs;
    double y_obs;
    double x_p;
    double y_p;
    double theta_p;
    //check the transformed observations for each particle
    for(int i=0;i<num_particles;i++){
        x_p = particles[i].x;
        y_p = particles[i].y;
        theta_p = particles[i].theta;
        
        //vector to store landmarks in range of the sensor
        vector<LandmarkObs> lmInRange;
        //find landmarks within sensor range
        for(int k=0;k<map_landmarks.landmark_list.size();k++){
            float x_lm = map_landmarks.landmark_list[k].x_f;
            float y_lm = map_landmarks.landmark_list[k].y_f;
            int id_lm = map_landmarks.landmark_list[k].id_i;
            
            if((fabs(x_p - x_lm) <= sensor_range) && (fabs(y_p - y_lm) <= sensor_range)){
                //cout << "lm id: " << id_lm << "lm x: " << x_lm << "lm y: " << y_lm << endl;
                lmInRange.push_back(LandmarkObs{ id_lm, x_lm, y_lm });
            }
        }
        //create vector to store observations in map coordinates
        vector<LandmarkObs> obsMap;
        double x_obs_map;
        double y_obs_map;
        
        //transform each observation
        for(int j=0;j<observations.size();j++){
            x_obs = observations[j].x;
            y_obs = observations[j].y;
            x_obs_map = x_obs*cos(theta_p) - y_obs*sin(theta_p) + x_p;
            y_obs_map = x_obs*sin(theta_p) + y_obs*cos(theta_p) + y_p;
            //cout << "x obs map: " << x_obs_map << " y obs map: " << y_obs_map << endl;
            //cout << "x_p: " << x_p << " y_p: " << y_p << endl;
            obsMap.push_back(LandmarkObs{ observations[j].id, x_obs_map, y_obs_map });
        }
        //Associate the predicted landmarks in range with the observations in map coordinate
        dataAssociation(lmInRange, obsMap);
        
        //reset particle weight
        particles[i].weight = 1.0;
        
        //set variables for updating weights
        double pi = M_PI;
        double sig_x = std_landmark[0];
        double sig_y = std_landmark[1];
        double weight_obs = 1.0;
        double x_lm;
        double y_lm;
        
        //loop through the weight calculated from each observation
        for(int l=0;l<obsMap.size();l++){
            x_obs_map = obsMap[l].x;
            y_obs_map = obsMap[l].y;
            int id_nearest_lm = obsMap[l].id;
            //cout << "nearest landmark id: " << id_nearest_lm << endl;
            
            //loop through to find matching closest observation
            for(int m=0;m<lmInRange.size();m++){
                //cout << "landmark ID: " << lmInRange[m].id << endl;
                if(lmInRange[m].id == id_nearest_lm){
                    x_lm = lmInRange[m].x;
                    y_lm = lmInRange[m].y;
                    //cout << "landmark_in x" << x_lm << endl;
                    //cout << "landmark_in y" << y_lm << endl;
                }
            }
            //cout << "landmark_out x" << x_lm << endl;
            //cout << "landmark_out y" << y_lm << endl;
            //calculate weight of particle
            double num =exp(-(pow(x_obs_map - x_lm,2)/(2*sig_x*sig_x) + pow(y_obs_map - y_lm,2)/(2*sig_y*sig_y)));
            double denum = 2*pi*sig_x*sig_y;
            weight_obs = num/denum;
            //cout << "weight obs: " << weight_obs << endl;
            particles[i].weight *= weight_obs;
        }
    }
    
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    vector<Particle> new_particles;
    
    vector<double> weights;
    
    //gather all the weights
    for(int i=0; i<num_particles; i++){
        weights.push_back(particles[i].weight);
    }
    double weights_sum = accumulate(weights.begin(),weights.end(),0);
    
    //normalize the weights
    for(int i=0; i<num_particles; i++){
        particles[i].weight = particles[i].weight/weights_sum;
    }
    
    random_device rd;
    mt19937 gen(rd());
    discrete_distribution<> d(weights.begin(), weights.end());
    map<int, int> m;
    for(int n=0; n<num_particles; ++n) {
        new_particles.push_back(particles[d(gen)]);
    }
    particles = new_particles;
    
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
    
    //Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();
    
    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
    
    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
    vector<int> v = best.associations;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
    vector<double> v = best.sense_x;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
    vector<double> v = best.sense_y;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
