if __name__ == '__main__':
    # Test code
    import os
    os.environ['PATH'] += "C:\\Program Files\\Java\\jre1.8.0_171\\bin\\server;"
    os.environ['CLASSPATH'] = "C:\\Users\\Luke\\Documents\\Dropbox\\Research\\PCG_2019\\SMBPCG\\simulator\\target\\classes;"

    import jnius

    SimulationHandler = jnius.autoclass('edu.unc.cs.smbpcg.simulator.SimulationHandler')
    LevelParser = jnius.autoclass('ch.idsia.mario.engine.level.LevelParser')
    level = LevelParser.createLevelASCII("..\\simulator\\mario-1-1.txt")
    handler = SimulationHandler(level)
    handler.invoke()
