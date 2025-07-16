/*
 * Copyright (c) 2025 Chloride Technology
 * GitHub: https://github.com/chloride-tech/
 * Website: https://91net.top/
 */
 
package tech.chloride;

import dev.backday.utils.OutputUtil;
import dev.backday.utils.TimerUtil;
import dev.backday.utils.math.MathUtils;

import javax.vecmath.Vector2f;

public class CalculateThread extends Thread {

    private static final float INITIAL_TEMPERATURE = 1000.0f;
    private static final float MIN_TEMPERATURE = 0.001f;
    private static final float COOLING_RATE = 0.995f;
    private static final int MAX_ITERATIONS = 10000;


    private static final float DISTANCE_WEIGHT_START = 0.3f;
    private static final float DISTANCE_WEIGHT_END = 0.8f;
    private static final float GRADIENT_STEP = 0.5f;
    private static final int GRADIENT_ITERATIONS = 20;
    private static final float LOCAL_SEARCH_MIN_STEP = 0.05f;

    private float temperature;
    private Vector2f currentSolution;
    private Vector2f bestSolution;
    private double currentEnergy;
    private double bestEnergy;
    private double bestDistance;
    private float currentDistanceWeight;
    public boolean stop;

    private final ProjectileUtil.EnderPearlPredictor predictor;
    private final boolean targetHitEntity;
    private final double startX, startY, startZ;

    public CalculateThread(double predictX, double predictY, double predictZ,
                           double minMotionY, double maxMotionY, boolean targetHitEntity) {
        this.predictor = new ProjectileUtil.EnderPearlPredictor(predictX, predictY, predictZ, minMotionY, maxMotionY);
        this.startX = predictX;
        this.startY = predictY;
        this.startZ = predictZ;
        this.targetHitEntity = targetHitEntity;
        this.temperature = INITIAL_TEMPERATURE;
        this.stop = false;
        this.currentDistanceWeight = DISTANCE_WEIGHT_START;


        this.currentSolution = new Vector2f(
                (float)MathUtils.getRandomInRange(-180, 180),
                (float)MathUtils.getRandomInRange(-90, 90)
        );
        this.bestSolution = new Vector2f(currentSolution);
    }

    @Override
    public void run() {
        TimerUtil timer = new TimerUtil();
        timer.reset();


        currentEnergy = evaluateSolution(currentSolution);
        bestEnergy = currentEnergy;
        bestDistance = calculateDistance(currentSolution);

        OutputUtil.log("Initial solution: yaw=" + currentSolution.x +
                ", pitch=" + currentSolution.y + ", distance=" + bestDistance);


        int iteration = 0;
        while (temperature > MIN_TEMPERATURE && !stop && iteration < MAX_ITERATIONS) {

            currentDistanceWeight = DISTANCE_WEIGHT_START +
                    (DISTANCE_WEIGHT_END - DISTANCE_WEIGHT_START) *
                            (1 - temperature / INITIAL_TEMPERATURE);


            Vector2f newSolution = generateHeuristicNeighbor(currentSolution);
            double newEnergy = evaluateSolution(newSolution);
            double newDistance = calculateDistance(newSolution);


            double energyDelta = (newEnergy - currentEnergy) *
                    (1 + currentDistanceWeight * (bestDistance - newDistance)/Math.max(bestDistance, 0.1));

            if (acceptSolution(energyDelta)) {
                currentSolution = new Vector2f(newSolution);
                currentEnergy = newEnergy;

                if (isBetterSolution(newEnergy, newDistance)) {
                    updateBestSolution(newSolution, newEnergy, newDistance);


                    if (bestEnergy >= 1.0) {
                        performGradientDescent();
                        break;
                    }
                }
            }

            temperature *= COOLING_RATE;
            iteration++;
        }


        if (!stop && bestEnergy >= 0.7) {
            performLocalRefinement();
        }

        OutputUtil.log(String.format("Iterations: %d, best solution: yaw=%.1f, pitch=%.1f, distance=%.1f",
                iteration, bestSolution.x, bestSolution.y, bestDistance));
    }


    private Vector2f generateHeuristicNeighbor(Vector2f current) {

        float yawChange = (float)(MathUtils.getRandomInRange(-temperature, temperature) * 18);
        float pitchChange = (float)(MathUtils.getRandomInRange(-temperature, temperature) * 9);


        if (bestDistance < Double.MAX_VALUE) {
            Vector2f gradient = estimateGradient(current);
            yawChange += gradient.x * temperature * 5;
            pitchChange += gradient.y * temperature * 5;
        }

        return new Vector2f(
                (float)MathUtils.clamp(current.x + yawChange, -180, 180),
                (float)MathUtils.clamp(current.y + pitchChange, -90, 90)
        );
    }


    private Vector2f estimateGradient(Vector2f point) {
        double baseDistance = calculateDistance(point);
        if (baseDistance == Double.MAX_VALUE) return new Vector2f(0.0f, 0.0f);


        Vector2f yawPlus = new Vector2f(point.x + GRADIENT_STEP, point.y);
        double yawPlusDist = calculateDistance(yawPlus);
        float yawGrad = (float)((yawPlusDist - baseDistance) / GRADIENT_STEP);


        Vector2f pitchPlus = new Vector2f(point.x, point.y + GRADIENT_STEP);
        double pitchPlusDist = calculateDistance(pitchPlus);
        float pitchGrad = (float)((pitchPlusDist - baseDistance) / GRADIENT_STEP);


        return new Vector2f(-yawGrad, -pitchGrad);
    }


    private void performGradientDescent() {
        Vector2f current = new Vector2f(bestSolution);
        double currentDist = bestDistance;
        float step = GRADIENT_STEP;

        for (int i = 0; i < GRADIENT_ITERATIONS && !stop; i++) {
            Vector2f gradient = estimateGradient(current);

            if (gradient.lengthSquared() < 0.0001f) break; 


            Vector2f newSolution = new Vector2f(
                    current.x - (step * gradient.x),
                    current.y - (step * gradient.y)
            );

            double newDist = calculateDistance(newSolution);
            if (newDist < currentDist) {
                current.set(newSolution);
                currentDist = newDist;

                if (newDist < bestDistance) {
                    bestSolution.set(current);
                    bestDistance = currentDist;
                    OutputUtil.log(String.format("Gradient descent to find a better solution: distance=%.1f", bestDistance));
                }
            } else {
                step *= 0.8f;
            }
        }
    }


    private void performLocalRefinement() {
        float step = 1.0f;
        int improvements = 0;

        while (step > LOCAL_SEARCH_MIN_STEP && improvements < 5 && !stop) {
            boolean improved = false;


            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    if (dx == 0 && dy == 0) continue;

                    Vector2f neighbor = new Vector2f(
                            bestSolution.x + dx * step,
                            bestSolution.y + dy * step
                    );

                    double newDist = calculateDistance(neighbor);
                    if (newDist < bestDistance) {
                        bestSolution.set(neighbor);
                        bestDistance = newDist;
                        improved = true;
                        OutputUtil.log(String.format("Search to find a better solution: distance=%.1f", bestDistance));
                    }
                }
            }

            if (improved) {
                improvements++;
            } else {
                step *= 0.5f;
            }
        }
    }


    private double evaluateSolution(Vector2f solution) {
        ProjectileUtil.ProjectileHit hit = predictor.assessSingleRotationHit(solution, 0);

        boolean validSide = hit.getLandingPosition() != null &&
                hit.getLandingPosition().sideHit.getIndex() == 1;

        if (!validSide) {
            return 0.0;
        }

        double baseScore = 0;
        if (targetHitEntity) {
            baseScore = hit.isHitEntity() ? 1.0 : hit.isHasLanded() ? 0.5 : 0.0;
        } else {
            baseScore = hit.isHasLanded() ? (hit.isHitEntity() ? 0.8 : 1.0) : 0.0;
        }


        double distance = calculateDistance(solution);
        double distanceFactor = 1.0 / (1.0 + distance / 5.0);

        return baseScore * (0.5 + 0.5 * distanceFactor);
    }


    private double calculateDistance(Vector2f solution) {
        ProjectileUtil.ProjectileHit hit = predictor.assessSingleRotationHit(solution, 0);
        if (!hit.isHasLanded()) {
            return Double.MAX_VALUE;
        }
        return Math.sqrt(
                Math.pow(hit.getPosX() - startX, 2) +
                        Math.pow(hit.getPosY() - startY, 2) +
                        Math.pow(hit.getPosZ() - startZ, 2)
        );
    }


    private boolean acceptSolution(double energyDelta) {
        if (energyDelta > 0) {
            return true; 
        }
        return Math.random() < Math.exp(energyDelta / temperature);
    }


    private boolean isBetterSolution(double newEnergy, double newDistance) {
        if (newEnergy > bestEnergy) return true;
        if (newEnergy == bestEnergy && newDistance < bestDistance) return true;
        return false;
    }


    private void updateBestSolution(Vector2f solution, double energy, double distance) {
        bestSolution.set(solution);
        bestEnergy = energy;
        bestDistance = distance;
        OutputUtil.log(String.format("Found a better solution: yaw=%.1f, pitch=%.1f, score=%.3f, distance=%.1f",
                bestSolution.x, bestSolution.y, bestEnergy, bestDistance));
    }

    public Vector2f getSolution() {
        return new Vector2f(bestSolution);
    }
}
