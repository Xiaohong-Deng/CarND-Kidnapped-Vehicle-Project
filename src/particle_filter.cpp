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
#include <cfloat>

#include "particle_filter.h"

#define ALMOST_ZERO 0.0001

using namespace std;

namespace {
    /* @param coord[] is Array of dimension 2 [x coordinate, y coordinate]
    * @param std[] is Array of dimension 2 [standard deviation of x, standard deviation of y]
    * @param mean[] is Array of dimension 2 [mean of x, mean of y]
    */
    double compPdf(double coord[], double std_landmark[], double mean[]) {
        double x, y, std_x, std_y, mu_x, mu_y;
        double pdf;

        x = coord[0];
        y = coord[1];
        std_x = std_landmark[0];
        std_y = std_landmark[1];
        mu_x = mean[0];
        mu_y = mean[1];

        double pdf_coef = 1 / (2 * M_PI * std_x * std_y);
        double expo_1 = (x - mu_x) * (x - mu_x) / (2 * std_x * std_x);
        double expo_2 = (y - mu_y) * (y - mu_y) / (2 * std_y * std_y);

        pdf = pdf_coef * exp(-(expo_1 + expo_2));

        return pdf;
    }

    /*  normalize the weights vector such that all elements sum to 1.
     *  @param a Vector containing weights of particles
    */
    void normWeights(vector<double> &weights) {
        double sum_of_weights = 0.0;

        for (auto &w: weights) {
            sum_of_weights += w;
        }

        for (auto &w: weights) {
            w = w / sum_of_weights;
        }
    }
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 10;
    default_random_engine gen;
    double std_x, std_y, std_theta;

    std_x = std[0];
    std_y = std[1];
    std_theta = std[2];

    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_theta);

    cout << "ground truth is " << x << " " << y << endl;

    particles.reserve(static_cast<unsigned long>(num_particles));
    weights.reserve(static_cast<unsigned long>(num_particles));
    for (int i = 0; i < num_particles; i++) {
        double sample_x, sample_y, sample_theta;

        sample_x = dist_x(gen);
        sample_y = dist_y(gen);
        sample_theta = dist_theta(gen);

        cout << "random particle coordinates are " << sample_x << " " << sample_y << endl;

        Particle p_ = {/* id */ i,
                        /* x */ sample_x,
                        /* y */ sample_y,
                        /* theta */ sample_theta,
                        /* weight */ 1.0};
        particles.push_back(p_);

        weights.push_back(p_.weight);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    default_random_engine gen;
    double std_x, std_y, std_theta;

    std_x = std_pos[0];
    std_y = std_pos[1];
    std_theta = std_pos[2];

    normal_distribution<double> dist_x(0.0, std_x);
    normal_distribution<double> dist_y(0.0, std_y);
    normal_distribution<double> dist_theta(0.0, std_theta);

    double c1, c2;
    bool has_yaw_rate = (fabs(yaw_rate) > ALMOST_ZERO);

    if (has_yaw_rate) {
        c1 = velocity / yaw_rate;
        c2 = yaw_rate * delta_t;
    } else {
        c1 = velocity * delta_t;
        c2 = 0.0;
    }

    for (auto &p : particles) {
        cout << "before prediction particle coordinate is " << p.x << " " << p.y << endl;
        double pred_x, pred_y, pred_theta;

        if (has_yaw_rate) {
            pred_x = p.x + c1 * (sin(p.theta + c2) - sin(p.theta));
            pred_y = p.y + c1 * (cos(p.theta) - cos(p.theta + c2));
            pred_theta = p.theta + c2;
        } else {
            pred_x = p.x + c1 * cos(p.theta);
            pred_y = p.y + c1 * sin(p.theta);
            pred_theta = p.theta;
        }

        p.x = pred_x + dist_x(gen);
        p.y = pred_y + dist_y(gen);
        p.theta = pred_theta + dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
    // LandmarkObs is defined in helper_functions.h. It doesn't have to be in vehicle coordinate system
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
    vector<double> new_weights;
    new_weights.reserve(static_cast<unsigned long>(num_particles));

    // step 1: associate landmarks to all the particles
    // keep track of landmark id, observed x, y coordinates in map coordinate system
    // compute weights for all the particles

    // each time we confirm a landmark by nearest neighbor, we assign its id, x and y coordinates to
    // particle. Also we add the weight contribution of this landmark to new_weight. Once all the observations
    // are associated for a particle, update its weight with new_weight. Once all particles are updated, assign
    // weights new_weights
    for (auto &p : particles) {
        cout << "predicted particle coordinate is " << p.x << " " << p.y << endl;
        double new_weight = 1.0;

        vector<int> new_associations;
        vector<double> new_sense_x;
        vector<double> new_sense_y;

        new_associations.reserve(observations.size());
        new_sense_x.reserve(observations.size());
        new_sense_y.reserve(observations.size());

        if (observations.empty()) {
            new_weight = 0.0;
            cout << "no observation!!!" << endl;
        } else {
            for (auto &ob : observations) {
                // landmark_id, obs_x, obs_y are for p.associations, p.sense_x and p.sense_y
                double map_x, map_y, landmark_x, landmark_y, min_distance;
                int landmark_id;

                map_x = p.x + cos(p.theta) * ob.x - sin(p.theta) * ob.y;
                map_y = p.y + sin(p.theta) * ob.x + cos(p.theta) * ob.y;
                landmark_x = DBL_MIN;
                landmark_y = DBL_MIN;
                min_distance = DBL_MAX;
                landmark_id = -1;

                for (auto &landmark : map_landmarks.landmark_list) {
                    if (dist(p.x, p.y, landmark.x_f, landmark.y_f) < sensor_range) {
                        double distance = dist(map_x, map_y, landmark.x_f, landmark.y_f);
                        if (distance < min_distance) {
                            min_distance = distance;
                            landmark_id = landmark.id_i;
                            landmark_x = landmark.x_f;
                            landmark_y = landmark.y_f;
                        }
                    }
                }

                // we traversed all the landmarks for a single observation (for a single particle), we add its associated landmark id
                // to p.associations, and update p.sense_x, p.sense_y
                new_associations.push_back(landmark_id);
                new_sense_x.push_back(map_x);
                new_sense_y.push_back(map_y);

                // compute pdf for the (landmark, observation) pair
                double pdf;

                double coordinates[] = {map_x, map_y};
                double means[] = {landmark_x, landmark_y};

                pdf = compPdf(coordinates, std_landmark, means);

                // update weight
                new_weight *= pdf;
            }
        }

        //weight updating is finished, assign particle weight the new weight
        new_weights.push_back(new_weight);
        p = SetAssociations(p, new_associations, new_sense_x, new_sense_y);
        cout << "Associations are " << getAssociations(p) << endl;
        cout << "SenseX are " << getSenseX(p) << endl;
        cout << "SenseY are " << getSenseY(p) << endl;
    }
    // we now have all the particle weights updated, time to normalize the weights
    normWeights(new_weights);
    weights = new_weights;

    // particle.weight and weights store the same things, I keep them both for now because they are provided code
    // I sync particle.weight but keep the sync code commented out. think it's not necessary. might delete particle.weight
    // later
    double sum_weights = 0.0;
    for (int i = 0; i < weights.size(); i++) {
        sum_weights += weights[i];
        particles[i].weight = weights[i];
    }
    cout << "weights sum to " << sum_weights << endl;
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    cout << "Resampling!!" << endl;
    random_device rd;
    mt19937 gen(rd());
    discrete_distribution<> index_generator(weights.begin(), weights.end());
    vector<Particle> new_samples;

    cout << "max weight is " << *max_element(weights.begin(), weights.end()) << endl;

    new_samples.reserve(static_cast<unsigned long>(num_particles));
    for (int i = 0; i < num_particles; i++) {
        Particle p_cp;
        // make a copy of particle
        p_cp = particles[index_generator(gen)];
        new_samples.push_back(p_cp);
        cout << "Sampled particles coord is (" << p_cp.x << ", " << p_cp.y << ")" << endl;
    }

    particles = new_samples;
    cout << "Resampled " << particles.size() << " particles!!" << endl;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
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
