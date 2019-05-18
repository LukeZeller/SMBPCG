package edu.unc.cs.smbpcg.simulator;

import ch.idsia.ai.agents.human.HumanKeyboardAgent;
import ch.idsia.mario.engine.level.Level;
import ch.idsia.mario.engine.level.LevelParser;
import ch.idsia.mario.simulation.BasicSimulator;
import ch.idsia.mario.simulation.Simulation;
import ch.idsia.tools.CmdLineOptions;
import ch.idsia.tools.EvaluationInfo;
import ch.idsia.tools.EvaluationOptions;
import ch.idsia.tools.ToolsConfigurator;

public class SimulationHandler {

    public static void main(String[] args) {
        Level level = LevelParser.createLevelASCII("mario-1-1.txt");
        SimulationHandler handler = new SimulationHandler(level);
        handler.invoke();
    }

    private Simulation simulation;
    private EvaluationOptions evaluationOptions;

    public SimulationHandler(Level level) {
        // Initialize default parameters for manual testing purposes
        // TODO: Generalize and implement for A* Agent invokation
        evaluationOptions = new CmdLineOptions(new String[]{""});
        evaluationOptions.setMaxFPS(false);
        evaluationOptions.setVisualization(true);
        evaluationOptions.setAgent(new HumanKeyboardAgent());
        evaluationOptions.setLevel(level);

        ToolsConfigurator.CreateMarioComponentFrame(evaluationOptions);
        simulation = new BasicSimulator(evaluationOptions.getSimulationOptionsCopy());
        simulation.setSimulationOptions(evaluationOptions);
    }

    public EvaluationInfo invoke() {
        return simulation.simulateOneLevel();
    }

}
