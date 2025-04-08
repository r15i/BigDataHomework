import org.apache.spark.SparkConf;
import org.apache.spark.api.java.*;
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

public class G39HW1 {

    public static void main(String[] args) {



        Logger.getRootLogger().setLevel(Level.OFF);
        Scanner scanner = new Scanner(System.in);
        SparkConf conf = new SparkConf().setAppName("KMeansClustering").setMaster("local");
        JavaSparkContext sc = new JavaSparkContext(conf);

        System.out.print("Enter The Number of Partitions: ");
        int L = scanner.nextInt();
        System.out.print("Enter The Number of clusters: ");
        int K = scanner.nextInt();
        System.out.print("Enter The Number of Lloyd's iterations: ");
        int iterations = scanner.nextInt();

        JavaRDD<String> lines = sc.textFile("uber_small.csv").repartition(L);
        JavaRDD<Tuple2<Vector, String>> groupRDD = lines.map(myMethods::LinesToInputPoints);
        List<Tuple2<String, Integer>> groupCounts = myMethods.CountEachGroup(groupRDD).collect();

        JavaRDD<Vector> pointsRDD = groupRDD.map(Tuple2::_1).cache();
        KMeansModel model = KMeans.train(pointsRDD.rdd(), K, iterations);
        Vector[] centroids = model.clusterCenters();

        double standardObjective = MRComputeStandardObjective(groupRDD, centroids);
        double fairObjective = MRComputeFairObjective(groupRDD, centroids);

        System.out.println("======= EXAMPLE OF OUTPUT FOR L = " + L + ", K = " + K + ", M = " + iterations + " =======\n");
        System.out.println("Input file = uber_small.csv, L = " + L + ", K = " + K + ", M = " + iterations);
        long N = groupRDD.count();
        long NA = groupCounts.stream().filter(t -> t._1().equals("A")).mapToLong(Tuple2::_2).sum();
        long NB = groupCounts.stream().filter(t -> t._1().equals("B")).mapToLong(Tuple2::_2).sum();
        System.out.println("N = " + N + ", NA = " + NA + ", NB = " + NB);
        System.out.printf("Delta(U,C) = %.6f\n", standardObjective);
        System.out.printf("Phi(A,B,C) = %.6f\n", fairObjective);

        Map<Tuple2<Integer, String>, Integer> centroidGroupCounts = groupRDD.mapToPair(point -> {
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
        }).reduceByKey(Integer::sum).collectAsMap();

        for (int i = 0; i < centroids.length; i++) {
            int countA = centroidGroupCounts.getOrDefault(new Tuple2<>(i, "A"), 0);
            int countB = centroidGroupCounts.getOrDefault(new Tuple2<>(i, "B"), 0);
            System.out.printf("i = %d, center = (%.6f,%.6f), NA%d = %d, NB%d = %d\n", i, centroids[i].apply(0), centroids[i].apply(1), i, countA, i, countB);
        }

        sc.close();
    }

    public static double MRComputeStandardObjective(JavaRDD<Tuple2<Vector, String>> groupRDD, Vector[] centroids) {
        JavaPairRDD<String, Double> distancesRDD = groupRDD.mapToPair(point -> myMethods.GetClosestDistance(point, centroids));
        long numberOfPoints = groupRDD.count();
        double totalSquaredDistance = distancesRDD.mapToDouble(Tuple2::_2).reduce(Double::sum);
        return totalSquaredDistance / numberOfPoints;
    }
    public static double MRComputeFairObjective(JavaRDD<Tuple2<Vector, String>> groupRDD, Vector[] centroids) {
        JavaPairRDD<String, Double> distancesRDD = groupRDD.mapToPair(point -> myMethods.GetClosestDistance(point, centroids));
        JavaPairRDD<String, Double> totalDistances = distancesRDD.reduceByKey(Double::sum);
        JavaPairRDD<String, Integer> groupCounts = myMethods.CountEachGroup(groupRDD);

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
    public static Tuple2<Vector, String> LinesToInputPoints(String line) {
        String[] parts = line.split(",");
        double lat = Double.parseDouble(parts[0]);
        double lon = Double.parseDouble(parts[1]);
        String group = parts[2];
        Vector point = Vectors.dense(lat, lon);
        return new Tuple2<>(point, group);
    }

    public static Tuple2<String, Double> GetClosestDistance(Tuple2<Vector, String> tuple, Vector[] centroids) {
        Vector point = tuple._1();
        double minDistance = Double.MAX_VALUE;
        for (Vector centroid : centroids) {
            double dist = Vectors.sqdist(point, centroid);
            if (dist < minDistance) {
                minDistance = dist;
            }
        }
        return new Tuple2<>(tuple._2(), minDistance);
    }

    public static JavaPairRDD<String, Integer> CountEachGroup(JavaRDD<Tuple2<Vector, String>> groupRDD) {
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
}