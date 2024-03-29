package ch.idsia.mario.engine;

import ch.idsia.ai.agents.Agent;
import ch.idsia.ai.agents.ai.RandomAgent;
import ch.idsia.ai.agents.human.CheaterKeyboardAgent;
import ch.idsia.mario.engine.level.LevelParser;
import ch.idsia.mario.engine.sprites.Mario;
import ch.idsia.mario.environments.Environment;
import ch.idsia.tools.EvaluationInfo;
import ch.idsia.tools.GameViewer;
import ch.idsia.tools.tcp.ServerAgent;

import javax.swing.*;
import java.awt.*;
import java.awt.event.FocusEvent;
import java.awt.event.FocusListener;
import java.awt.event.KeyAdapter;
import java.awt.image.VolatileImage;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import edu.unc.cs.smbpcg.simulator.KeyPress;
import edu.unc.cs.smbpcg.simulator.MoveList;
import reader.JsonReader;


public class MarioComponent extends JComponent implements Runnable, /*KeyListener,*/ FocusListener, Environment {
    private static final long serialVersionUID = 790878775993203817L;
    public static final int TICKS_PER_SECOND = 24;

    private boolean running = false;
    private int width, height;
    private GraphicsConfiguration graphicsConfiguration;
    private Scene scene;
    private boolean focused = false;

    int frame;
    int delay;
    Thread animator;

    private int ZLevelEnemies = 1;
    private int ZLevelScene = 1;

    public void setGameViewer(GameViewer gameViewer) {
        this.gameViewer = gameViewer;
    }

    private GameViewer gameViewer = null;

    private Agent agent = null;
    private CheaterKeyboardAgent cheatAgent = null;

    private KeyAdapter prevHumanKeyBoardAgent;
    private LevelScene levelScene = null;

    public MarioComponent(int width, int height) {
        adjustFPS();

        this.setFocusable(true);
        this.setEnabled(true);
        this.width = width;
        this.height = height;

        Dimension size = new Dimension(width, height);

        setPreferredSize(size);
        setMinimumSize(size);
        setMaximumSize(size);

        setFocusable(true);

        if (this.cheatAgent == null)
        {
            this.cheatAgent = new CheaterKeyboardAgent();
            this.addKeyListener(cheatAgent);
        }        

        GlobalOptions.registerMarioComponent(this);
    }

    public void adjustFPS() {
        int fps = GlobalOptions.FPS;
        delay = (fps > 0) ? (fps >= GlobalOptions.InfiniteFPS) ? 0 : (1000 / fps) : 100;
//        System.out.println("Delay: " + delay);
    }

    public void paint(Graphics g) {
    }

    public void update(Graphics g) {
    }

    public void init() {
        graphicsConfiguration = getGraphicsConfiguration();
//        if (graphicsConfiguration != null) {
            Art.init(graphicsConfiguration);
//        }
    }

    public void start() {
        if (!running) {
            running = true;
            animator = new Thread(this, "Game Thread");
            animator.start();
        }
    }

    public void stop() {
        running = false;
    }

    public void run() {

    }

    public EvaluationInfo run1(int currentTrial, int totalNumberOfTrials) {
        running = true;
        adjustFPS();
        EvaluationInfo evaluationInfo = new EvaluationInfo();

        VolatileImage image = null;
        Graphics g = null;
        Graphics og = null;

        image = createVolatileImage(320, 240);
        g = getGraphics();
        og = image.getGraphics();

        if (!GlobalOptions.VisualizationOn) {
            String msgClick = "Vizualization is not available";
            drawString(og, msgClick, 160 - msgClick.length() * 4, 110, 1);
            drawString(og, msgClick, 160 - msgClick.length() * 4, 110, 7);
        }

        addFocusListener(this);

        // Remember the starting time
        long tm = System.currentTimeMillis();
        long tick = tm;
        int marioStatus = Mario.STATUS_RUNNING;
        this.levelScene.mario.setMode(Mario.MODE.MODE_SMALL);

        int totalActionsPerfomed = 0;
        int jumpActionsPerformed = 0;
        int hardJumpActionsPerformed = 0;
        int mediumJumpActionsPerformed = 0;
        int easyJumpActionsPerformed = 0;
        int trivialJumpActionsPerformed = 0;
        int previousJumpFrame = -1;
        frame = 0;

        boolean marioDiedToFall = false;
        boolean marioDiedToEnemy = false;
        boolean marioRanOutOfTime = false;

        MoveList marioMoves = new MoveList();
        levelScene.mario.resetCoins();
        while (running) {
            scene.tick();
            if (gameViewer != null && gameViewer.getContinuousUpdatesState())
                gameViewer.tick();

            float alpha = 0;

            if (GlobalOptions.VisualizationOn) {
                og.fillRect(0, 0, 320, 240);
                scene.render(og, alpha);
            }

            if (agent instanceof ServerAgent && !((ServerAgent) agent).isAvailable()) {
                System.err.println("Agent became unavailable. Simulation Stopped");
                running = false;
                break;
            }

            float current_mario_x = levelScene.mario.x;
            float current_mario_y = levelScene.mario.y;

            if (current_mario_y > EvaluationInfo.lowestValidPosition && levelScene.getTimeLeft() > 0)  {
                marioDiedToFall = true;
            }
            if (!marioDiedToFall && levelScene.mario.getStatus() == 0 && levelScene.getTimeLeft() > 0)  {
                marioDiedToEnemy = true;
            }

            KeyPress kp = new KeyPress(agent.getAction(this/*DummyEnvironment*/));
            if (kp.isValid())
            {
                marioMoves.addKeyPress(kp);
                for (int button = 0; button < Environment.numberOfButtons; ++button){
                    if (kp.isPressed(button))
                    {
                        if(button == Mario.KEY_JUMP) {
                            /*
                             * If gapBetweenJumps is <= 1, that means the jump key was just held or pressed for the first time
                             * We categorize the difficulty of jumps based on # of frames between the end of one jump action
                             * which corresponds to the jump key being released and the start of the next jump action
                             * which corresponds to the jump key being pressed
                             */
                            int gapBetweenJumps = frame - previousJumpFrame;
                            if (gapBetweenJumps > 1)
                            {
                                jumpActionsPerformed++;
                                if (gapBetweenJumps <= EvaluationInfo.hardJumpThreshold) {
                                    hardJumpActionsPerformed++;
                                }
                                else if (gapBetweenJumps <= EvaluationInfo.mediumJumpThreshold) {
                                    mediumJumpActionsPerformed++;
                                }
                                else if (gapBetweenJumps <= EvaluationInfo.easyJumpThreshold) {
                                    easyJumpActionsPerformed++;
                                }
                                else if (gapBetweenJumps <= EvaluationInfo.trivialJumpThreshold) {
                                    trivialJumpActionsPerformed++;
                                }
                                else {
                                    /* there should never be a level large enough for the gap between two jumps to be larger
                                     * than the trivial jump threshold */
                                    throw new RuntimeException("Gap between jumps is impossibly big");
                                }
                            }
                            previousJumpFrame = frame;
                        }
                        ++totalActionsPerfomed;
                        // break;
                    }
                }
            }
            else
            {
                System.err.println("Null Action received. Skipping simulation...");
                stop();
            }

            if (hardJumpActionsPerformed + mediumJumpActionsPerformed + easyJumpActionsPerformed + trivialJumpActionsPerformed != jumpActionsPerformed) {
                throw new RuntimeException("Expected the total number of jumps to be equal to the sum of hard/medium/easy/trivial jumps\n" +
                        "Found " + jumpActionsPerformed + " jumps\n" +
                        " and have " + hardJumpActionsPerformed + " hard jumps\n" +
                        mediumJumpActionsPerformed + " medium jumps\n" +
                        easyJumpActionsPerformed + " easy jumps\n" +
                        trivialJumpActionsPerformed + " trivial jumps");
            }

            ((LevelScene) scene).mario.keys = kp.getPressed();
            ((LevelScene) scene).mario.cheatKeys = cheatAgent.getAction(null);

            if (GlobalOptions.VisualizationOn) {

                String msg = "Agent: " + agent.getName();
                ((LevelScene) scene).drawStringDropShadow(og, msg, 0, 7, 5);

                msg = "Selected Actions: ";
                ((LevelScene) scene).drawStringDropShadow(og, msg, 0, 8, 6);

                msg = "";
                if (kp.isValid())
                {
                    for (int i = 0; i < Environment.numberOfButtons; ++i)
                        msg += (kp.isPressed(i)) ? scene.keysStr[i] : "      ";
                }
                else
                    msg = "NULL";
                drawString(og, msg, 6, 78, 1);

                if (!this.hasFocus() && tick / 4 % 2 == 0) {
                    String msgClick = "CLICK TO PLAY";
//                    og.setColor(Color.YELLOW);
//                    og.drawString(msgClick, 320 + 1, 20 + 1);
                    drawString(og, msgClick, 160 - msgClick.length() * 4, 110, 1);
                    drawString(og, msgClick, 160 - msgClick.length() * 4, 110, 7);
                }
                og.setColor(Color.DARK_GRAY);
                ((LevelScene) scene).drawStringDropShadow(og, "FPS: ", 32, 2, 7);
                ((LevelScene) scene).drawStringDropShadow(og, ((GlobalOptions.FPS > 99) ? "\\infty" : GlobalOptions.FPS.toString()), 32, 3, 7);

                msg = totalNumberOfTrials == -2 ? "" : currentTrial + "(" + ((totalNumberOfTrials == -1) ? "\\infty" : totalNumberOfTrials) + ")";

                ((LevelScene) scene).drawStringDropShadow(og, "Trial:", 33, 4, 7);
                ((LevelScene) scene).drawStringDropShadow(og, msg, 33, 5, 7);

                if (width != 320 || height != 240) {
                    g.drawImage(image, 0, 0, 640 * 2, 480 * 2, null);
                } else {
                    g.drawImage(image, 0, 0, null);
                }
            } else {
                // Win or Die without renderer!! independently.
                marioStatus = ((LevelScene) scene).mario.getStatus();
                if (marioStatus != Mario.STATUS_RUNNING)
                    stop();
            }
            // Delay depending on how far we are behind.
            if (delay > 0)
                try {
                    tm += delay;
                    Thread.sleep(Math.max(0, tm - System.currentTimeMillis()));
                } catch (InterruptedException e) {
                    break;
                }
            // Advance the frame
            frame++;
        }
        // Remove redundant blank actions at the end of Mario's moves
        marioMoves.rstrip(new KeyPress());

        marioRanOutOfTime = levelScene.getTimeLeft() <= 0;
//=========
        evaluationInfo.agentType = agent.getClass().getSimpleName();
        evaluationInfo.agentName = agent.getName();
        evaluationInfo.marioStatus = levelScene.mario.getStatus();
        evaluationInfo.livesLeft = levelScene.mario.lives;
        evaluationInfo.lengthOfLevelPassedPhys = levelScene.mario.x;
        evaluationInfo.lengthOfLevelPassedCells = levelScene.mario.mapX;
        evaluationInfo.totalLengthOfLevelCells = levelScene.level.getWidthCells();
        evaluationInfo.totalLengthOfLevelPhys = levelScene.level.getWidthPhys();
        evaluationInfo.timeSpentOnLevel = levelScene.getStartTime();
        evaluationInfo.timeLeft = levelScene.getTimeLeft();
        evaluationInfo.totalTimeGiven = levelScene.getTotalTime();
        evaluationInfo.numberOfGainedCoins = levelScene.mario.coins;
//        evaluationInfo.totalNumberOfCoins   = -1 ; // TODO: total Number of coins.
        evaluationInfo.totalActionsPerformed = totalActionsPerfomed; // Counted during the play/simulation process
        evaluationInfo.jumpActionsPerformed = jumpActionsPerformed; // Counted during play/simulation
        evaluationInfo.hardJumpActionsPerformed = hardJumpActionsPerformed; // Counted during play/simulation
        evaluationInfo.mediumJumpActionsPerformed = mediumJumpActionsPerformed; // Counted during play/simulation
        evaluationInfo.easyJumpActionsPerformed = easyJumpActionsPerformed; // Counted during play/simulation
        evaluationInfo.trivialJumpActionsPerformed = trivialJumpActionsPerformed; // Counted during play/simulation

        evaluationInfo.totalFrames = frame;
        evaluationInfo.marioDiedToFall = marioDiedToFall;
        evaluationInfo.marioDiedToEnemy = marioDiedToEnemy;
        evaluationInfo.marioRanOutOfTime = marioRanOutOfTime;
        evaluationInfo.marioMode = levelScene.mario.getMode();
        evaluationInfo.marioMoves = marioMoves;
        evaluationInfo.killsTotal = levelScene.mario.world.killedCreaturesTotal;
//        evaluationInfo.Memo = "Number of attempt: " + Mario.numberOfAttempts;
        if (agent instanceof ServerAgent && levelScene.mario.keys != null /*this will happen if client quits unexpectedly in case of Server mode*/)
            ((ServerAgent)agent).integrateEvaluationInfo(evaluationInfo);
        return evaluationInfo;
    }

    private void drawString(Graphics g, String text, int x, int y, int c) {
        char[] ch = text.toCharArray();
        for (int i = 0; i < ch.length; i++) {
            g.drawImage(Art.font[ch[i] - 32][c], x + i * 8, y, null);
        }
    }

    /**
     * Method we added to directly take a Level instance and create it
     * 
     * Many of these leftover parameters actually seem to be irrelevant when we directly specify the level
     * @param seed
     * @param difficulty
     * @param type
     * @param levelLength
     * @param timeLimit
     * @param level
     */
    public void startLevel(long seed, int difficulty, int type, int levelLength, int timeLimit, ch.idsia.mario.engine.level.Level level) {
        scene = new LevelScene(graphicsConfiguration, this, seed, difficulty, type, levelLength, timeLimit);
        levelScene = ((LevelScene) scene);
        scene.init(level);
    }
    
    /**
     * Method we added to generate a level based on a json file that has been supplied
     * as a command line parameter
	 *
     * Many of these leftover parameters actually seem to be irrelevant when we directly specify the level
	 *
     * @param seed
     * @param difficulty
     * @param type
     * @param levelLength
     * @param timeLimit
     * @param filename
     * @param index
     */
    public void startLevel(long seed, int difficulty, int type, int levelLength, int timeLimit, String filename, int index) {
        scene = new LevelScene(graphicsConfiguration, this, seed, difficulty, type, levelLength, timeLimit);
        levelScene = ((LevelScene) scene);
        JsonReader reader = new JsonReader(filename);
        List<List<Integer>> input = reader.getLevel(index);
        LevelParser parser = new LevelParser();
        ch.idsia.mario.engine.level.Level level = parser.createLevelJson(input);
        //ch.idsia.mario.engine.level.Level level = parser.test();
        scene.init(level);
    }
    
    public void startLevel(long seed, int difficulty, int type, int levelLength, int timeLimit) {
        scene = new LevelScene(graphicsConfiguration, this, seed, difficulty, type, levelLength, timeLimit);
        levelScene = ((LevelScene) scene);
        scene.init();
        
        /**
         * From Jacob Schrum
         *
         * To output a useful text representation of whatever level
         * is generated, uncomment the code below and add the appropriate
         * import statements.
         */
//      try {
//          levelScene.level.saveText(new PrintStream(new FileOutputStream("TestLevel.txt")));
//      } catch (IOException e) {
//          // TODO Auto-generated catch block
//          e.printStackTrace();
//      }

    }

    public void levelFailed() {
//        scene = mapScene;
        levelScene.mario.lives--;
        stop();
    }

    public void focusGained(FocusEvent arg0) {
        focused = true;
    }

    public void focusLost(FocusEvent arg0) {
        focused = false;
    }

    public void levelWon() {
        stop();
//        scene = mapScene;
//        mapScene.levelWon();
    }

    public void toTitle() {
//        Mario.resetStatic();
//        scene = new TitleScene(this, graphicsConfiguration);
//        scene.init();
    }

    public List<String> getTextObservation(boolean Enemies, boolean LevelMap, boolean Complete, int ZLevelMap, int ZLevelEnemies) {
        if (scene instanceof LevelScene)
            return ((LevelScene) scene).LevelSceneAroundMarioASCII(Enemies, LevelMap, Complete, ZLevelMap, ZLevelEnemies);
        else {
            return new ArrayList<String>();
        }
    }

    public String getBitmapEnemiesObservation()
    {
        if (scene instanceof LevelScene)
            return ((LevelScene) scene).bitmapEnemiesObservation(1);
        else {
            //
            return new String();
        }                
    }

    public String getBitmapLevelObservation()
    {
        if (scene instanceof LevelScene)
            return ((LevelScene) scene).bitmapLevelObservation(1);
        else {
            //
            return null;
        }
    }

    // Chaning ZLevel during the game on-the-fly;
    public byte[][] getMergedObservationZ(int zLevelScene, int zLevelEnemies) {
        if (scene instanceof LevelScene)
            return ((LevelScene) scene).mergedObservation(zLevelScene, zLevelEnemies);
        return null;
    }

    public byte[][] getLevelSceneObservationZ(int zLevelScene) {
        if (scene instanceof LevelScene)
            return ((LevelScene) scene).levelSceneObservation(zLevelScene);
        return null;
    }

    public byte[][] getEnemiesObservationZ(int zLevelEnemies) {
        if (scene instanceof LevelScene)
            return ((LevelScene) scene).enemiesObservation(zLevelEnemies);
        return null;
    }

    public int getKillsTotal() {
        return levelScene.mario.world.killedCreaturesTotal;
    }

    public int getKillsByFire() {
        return levelScene.mario.world.killedCreaturesByFireBall;
    }

    public int getKillsByStomp() {
        return levelScene.mario.world.killedCreaturesByStomp;
    }

    public int getKillsByShell() {
        return levelScene.mario.world.killedCreaturesByShell;
    }

    public boolean canShoot() {
        return false;  //To change body of implemented methods use File | Settings | File Templates.
    }

    public byte[][] getCompleteObservation() {
        if (scene instanceof LevelScene)
            return ((LevelScene) scene).mergedObservation(this.ZLevelScene, this.ZLevelEnemies);
        return null;
    }

    public byte[][] getEnemiesObservation() {
        if (scene instanceof LevelScene)
            return ((LevelScene) scene).enemiesObservation(this.ZLevelEnemies);
        return null;
    }

    public byte[][] getLevelSceneObservation() {
        if (scene instanceof LevelScene)
            return ((LevelScene) scene).levelSceneObservation(this.ZLevelScene);
        return null;
    }

    public boolean isMarioOnGround() {
        return levelScene.mario.isOnGround();
    }

    public boolean mayMarioJump() {
        return levelScene.mario.mayJump();
    }

    public void setAgent(Agent agent) {
        this.agent = agent;
        if (agent instanceof KeyAdapter) {
            if (prevHumanKeyBoardAgent != null)
                this.removeKeyListener(prevHumanKeyBoardAgent);
            this.prevHumanKeyBoardAgent = (KeyAdapter) agent;
            this.addKeyListener(prevHumanKeyBoardAgent);
        }
    }

    public void setMarioInvulnerable(boolean invulnerable)
    {
        levelScene.mario.isMarioInvulnerable = invulnerable;
    }

    public void setPaused(boolean paused) {
        levelScene.paused = paused;
    }

    public void setZLevelEnemies(int ZLevelEnemies) {
        this.ZLevelEnemies = ZLevelEnemies;
    }

    public void setZLevelScene(int ZLevelScene) {
        this.ZLevelScene = ZLevelScene;
    }

    public float[] getMarioFloatPos()
    {
        return new float[]{this.levelScene.mario.x, this.levelScene.mario.y};
    }

    public float[] getEnemiesFloatPos()
    {
        if (scene instanceof LevelScene)
            return ((LevelScene) scene).enemiesFloatPos();
        return null;
    }

    public int getMarioMode()
    {
        return levelScene.mario.getMode();
    }

    public boolean isMarioCarrying()
    {
        return levelScene.mario.carried != null;
    }
}