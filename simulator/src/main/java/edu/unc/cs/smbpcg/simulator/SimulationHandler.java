package edu.unc.cs.smbpcg.simulator;

import ch.idsia.ai.agents.Agent;
import ch.idsia.ai.agents.human.HumanKeyboardAgent;
import ch.idsia.mario.engine.GlobalOptions;
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
        handler.setAgent(new HumanKeyboardAgent());
        System.out.println(handler.evaluationOptions.getAgent().getClass());
        handler.init();
        handler.invoke();
    }

    private Simulation simulation;
    private EvaluationOptions evaluationOptions;

    public SimulationHandler() {
        evaluationOptions = new CmdLineOptions(new String[]{""});
    }

    public SimulationHandler(Level level) {
        this();
        setLevel(level);
    }

    public void setAgent(Agent agent) {
        evaluationOptions.setAgent(agent);
    }

    public Level getLevel() {
        return evaluationOptions.getLevel();
    }

    public void setLevel(Level level) {
        evaluationOptions.setLevel(level);
    }

    public boolean isMaxFPS() {
        return evaluationOptions.isMaxFPS();
    }

    public void setMaxFPS(boolean isMaxFPS) {
        evaluationOptions.setMaxFPS(isMaxFPS);
    }

    public boolean getVisualization() {
        return evaluationOptions.isVisualization();
    }

    /* Due to somewhat cluttered design of Java simulator codebase, visualization
    status is stored and received from two different places in different components
    of the code:
        - Static global variable GlobalOptions.VisualizationOn
        - Instance property evaluationOptions.isVisualization()
    Both are used for certain drawing activities but are not guaranteed to be in sync
    (e.g. GlobalOptions.VisualizationOn is updated when BasicSimulator is instantiated
    but not when evaluationOptions.setVisualization(...) is called). For that reason,
    this setter proactively updates both so that global state of our simulation remains
    consistent.
     */
    public void setVisualization(boolean isVisualization) {
        GlobalOptions.VisualizationOn = isVisualization;
        evaluationOptions.setVisualization(isVisualization);
    }

    public void init() {
        ToolsConfigurator.CreateMarioComponentFrame(evaluationOptions);
        simulation = new BasicSimulator(evaluationOptions);
    }

    public EvaluationInfo invoke() {
        return simulation.simulateOneLevel();
    }

}