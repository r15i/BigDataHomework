import org.apache.spark.SparkConf;
import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

import org.apache.log4j.Logger;
import org.apache.log4j.Level;


import java.util.List;

public class K_Means_Clustering {

    public static void main(String[] args) {
        // Suppress Spark logs
        // useful if u dont care about the long infos from the spark service
        Logger.getRootLogger().setLevel(Level.OFF);

        SparkConf conf = new SparkConf().setAppName("KMeansClustering").setMaster("local");
        JavaSparkContext sc = new JavaSparkContext(conf); // connection to Spark

        JavaRDD<String> lines = sc.textFile("uber_small.csv"); // each line becomes a single element in RDD

        JavaRDD<Tuple2<Vector, String>> groupRDD = lines.map(
            new Function<String, Tuple2<Vector, String>>() {
                public Tuple2<Vector, String> call(String line) {
                    String[] parts = line.split(",");
                    double lat = Double.parseDouble(parts[0]);
                    double lon = Double.parseDouble(parts[1]);
                    String group = parts[2];
                    Vector point = Vectors.dense(lat, lon);
                    return new Tuple2<>(point, group);
                }
            }
        );

        // print some elements
         for (Tuple2<Vector, String> tuple : groupRDD.take(5)) {
             System.out.println("Point: " + tuple._1 + " Group: " + tuple._2);
         }

        // Extract points as RDD<Vector> for KMeans
        JavaRDD<Vector> pointsRDD = groupRDD.map(Tuple2::_1).cache();

        // KMeans parameters
        int K = 3;  // Number of clusters
        int iterations = 10;  // Lloyd's iterations

        // Train the KMeans model using Spark's built-in implementation
        KMeansModel model = KMeans.train(pointsRDD.rdd(), K, iterations);

        // Get the computed centroids
        Vector[] centroids = model.clusterCenters();

        // Compute and print standard objective function Δ(U,C)
        double standardObjective = MRComputeStandardObjective(groupRDD, centroids);
        System.out.println("Standard Objective Function Value: " + standardObjective);

        // Compute and print fair objective function Φ(A,B,C)
        double fairObjective = MRComputeFairObjective(groupRDD, centroids);
        System.out.println("Fair Objective Function Value: " + fairObjective);

        sc.close();
    }



    public static double MRComputeStandardObjective(JavaRDD<Tuple2<Vector, String>> groupRDD, Vector[] centroids) {
        // Calculate squared distance for each point to the nearest centroid
        // and maps each point to it's distance
        JavaPairRDD<String, Double> distancesRDD = groupRDD.mapToPair(
                new PairFunction<Tuple2<Vector, String>, String, Double>() {
                    public Tuple2<String, Double> call(Tuple2<Vector, String> tuple) {
                        Vector point = tuple._1();

                        // Find the closest centroid
                        double minDistance = Double.MAX_VALUE;
                        for (Vector centroid : centroids) {
                            double dist = Vectors.sqdist(point, centroid);
                            if (dist < minDistance) {
                                minDistance = dist;
                            }
                        }
                        return new Tuple2<>(tuple._2(), minDistance);
                    }
                });

        // Compute total squared distance for all points (ignoring the group)
        // does reducing part (maps each distances to the sum )
        double totalSquaredDistance = distancesRDD.mapToDouble(Tuple2::_2).reduce((a, b) -> a + b);

        return totalSquaredDistance;
    }



    /**
     * Computes the fair k-means objective function Φ(A, B, C).
     */
    public static double MRComputeFairObjective(JavaRDD<Tuple2<Vector, String>> groupRDD, Vector[] centroids) {
        // Compute (squared distance, group) for each point
        JavaPairRDD<String, Double> distancesRDD = groupRDD.mapToPair(
            new PairFunction<Tuple2<Vector, String>, String, Double>() {
                public Tuple2<String, Double> call(Tuple2<Vector, String> tuple) {
                    Vector point = tuple._1();
                    String group = tuple._2();

                    // Find closest centroid
                    double minDistance = Double.MAX_VALUE;
                    for (Vector centroid : centroids) {
                        double dist = Vectors.sqdist(point, centroid);
                        if (dist < minDistance) {
                            minDistance = dist;
                        }
                    }
                    return new Tuple2<>(group, minDistance);
                }
            });

        // Compute total squared distance for each group
        JavaPairRDD<String, Double> totalDistances = distancesRDD.reduceByKey(
            new Function2<Double, Double, Double>() {
                public Double call(Double a, Double b) {
                    return a + b;
                }
            });

        // Count number of points in each group
        JavaPairRDD<String, Integer> groupCounts = groupRDD.mapToPair(
            new PairFunction<Tuple2<Vector, String>, String, Integer>() {
                public Tuple2<String, Integer> call(Tuple2<Vector, String> tuple) {
                    return new Tuple2<>(tuple._2(), 1);
                }
            }).reduceByKey(
            new Function2<Integer, Integer, Integer>() {
                public Integer call(Integer a, Integer b) {
                    return a + b;
                }
            });

        // Collect results
        List<Tuple2<String, Double>> totalDistList = totalDistances.collect();
        List<Tuple2<String, Integer>> countList = groupCounts.collect();

        double fairObjectiveA = 0;
        double fairObjectiveB = 0;

        for (Tuple2<String, Double> distTuple : totalDistList) {
            for (Tuple2<String, Integer> countTuple : countList) {
                if (distTuple._1().equals(countTuple._1()) && countTuple._2() > 0) {
                    double avgDist = distTuple._2() / countTuple._2();
                    if (distTuple._1().equals("A")) {
                        fairObjectiveA = avgDist;
                    } else if (distTuple._1().equals("B")) {
                        fairObjectiveB = avgDist;
                    }
                }
            }
        }

        return Math.max(fairObjectiveA, fairObjectiveB);
    }
}


    
