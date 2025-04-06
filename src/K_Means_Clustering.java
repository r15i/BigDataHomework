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

import java.util.*;

import org.apache.log4j.Logger;
import org.apache.log4j.Level;

public class K_Means_Clustering {

    public static void main(String[] args) {
        // Suppress Spark logs
        // useful if u dont care about the long infos from the spark service
        Logger.getRootLogger().setLevel(Level.OFF);
        Scanner scanner = new Scanner(System.in); // READ INPUT
        SparkConf conf = new SparkConf().setAppName("KMeansClustering").setMaster("local");
        JavaSparkContext sc = new JavaSparkContext(conf); // connection to Spark
        // Number of Partitions
        System.out.print("Enter The Number of Partitions: ");
        int L = scanner.nextInt();
        System.out.print("Enter The Number of clusters: ");
        int K = scanner.nextInt();
        System.out.print("Enter The Number of Lloyd's iterations: ");
        int iterations = scanner.nextInt();
        JavaRDD<String> lines = sc.textFile("uber_small.csv").repartition(L); // each line becomes a single element in RDD
        JavaRDD<Tuple2<Vector, String>> groupRDD = lines.map(
                myMethods::LinesToInputPoints
        );
        List<Tuple2<String, Integer>> groupCounts = myMethods.CountEachGroup(groupRDD).collect();
        System.out.println("N => "+ groupRDD.count());
        for (Tuple2<String, Integer> entry : groupCounts) {
            System.out.println("N"+ entry._1() + " => " + entry._2());
        }
        // Extract points as RDD<Vector> for KMeans
        JavaRDD<Vector> pointsRDD = groupRDD.map(Tuple2::_1).cache();

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
        myMethods.MRPrintStatistics(groupRDD, centroids);
        sc.close();
    }



    public static double MRComputeStandardObjective(JavaRDD<Tuple2<Vector, String>> groupRDD, Vector[] centroids) {
        // Calculate squared distance for each point to the nearest centroid
        // and maps each point to it's distance
        JavaPairRDD<String, Double> distancesRDD = groupRDD.mapToPair(point ->{
            return myMethods.GetClosestDistance(point,centroids);});
        // Compute total squared distance for all points (ignoring the group)
        // does reducing part (maps each distances to the sum )
        long numberOfPoints=groupRDD.count();
        double totalSquaredDistance = distancesRDD.mapToDouble(Tuple2::_2).reduce((a, b) -> a + b);

        return totalSquaredDistance/numberOfPoints;
    }



    /**
     * Computes the fair k-means objective function Φ(A, B, C).
     */
    public static double MRComputeFairObjective(JavaRDD<Tuple2<Vector, String>> groupRDD, Vector[] centroids) {
        // Compute (squared distance, group) for each point
        JavaPairRDD<String, Double> distancesRDD = groupRDD.mapToPair(point ->{
            return myMethods.GetClosestDistance(point,centroids);});

        // Compute total squared distance for each group
        JavaPairRDD<String, Double> totalDistances = distancesRDD.reduceByKey(
            new Function2<Double, Double, Double>() {
                public Double call(Double a, Double b) {
                    return a + b;
                }
            });

        // Count number of points in each group
        JavaPairRDD<String, Integer> groupCounts = myMethods.CountEachGroup(groupRDD);

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
class myMethods {
    public static Tuple2<Vector, String> LinesToInputPoints(String line){
            String[] parts = line.split(",");
            double lat = Double.parseDouble(parts[0]);
            double lon = Double.parseDouble(parts[1]);
            String group = parts[2];
            Vector point = Vectors.dense(lat, lon);
            return new Tuple2<>(point, group);
    }
    public static Tuple2<String, Double> GetClosestDistance(Tuple2<Vector, String> tuple,Vector[] centroids){
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
    public static JavaPairRDD<String, Integer> CountEachGroup(JavaRDD<Tuple2<Vector, String>> groupRDD){
        return groupRDD.mapToPair(
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
    }
    public static void MRPrintStatistics(JavaRDD<Tuple2<Vector, String>> groupRDD, Vector[] centroids) {
        // (centroidIndex, group) -> 1
        JavaPairRDD<Tuple2<Integer, String>, Integer> groupCountsPerCentroid = groupRDD.mapToPair(point -> {
            Vector v = point._1();
            String group = point._2();
            double minDistance = Double.MAX_VALUE;
            int closestIndex = -1;

            for (int i = 0; i < centroids.length; i++) {
                double dist = Vectors.sqdist(v, centroids[i]);
                if (dist < minDistance) {
                    minDistance = dist;
                    closestIndex = i;
                }
            }

            return new Tuple2<>(new Tuple2<>(closestIndex, group), 1);
        }).reduceByKey(Integer::sum);

        // group results by centroid
        Map<Tuple2<Integer, String>, Integer> result = groupCountsPerCentroid.collectAsMap();

        for (int i = 0; i < centroids.length; i++) {
            int countA = result.getOrDefault(new Tuple2<>(i, "A"), 0);
            int countB = result.getOrDefault(new Tuple2<>(i, "B"), 0);
            System.out.println("Centroid c" + i + " => NA: " + countA + ", NB: " + countB);
        }
    }
}


    
