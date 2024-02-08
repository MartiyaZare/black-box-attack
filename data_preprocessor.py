from MZPackage.DataPreprocessor import DataPreprocessor

dp = DataPreprocessor()

# to convert YM dataset from dat to mat
#dp.datToMatBatch(srcFolder="Raw_Data/YM/diff/diff_0/data/diff_0-1/HypProc_1", dstFolder="Raw_Data/YM_mat/1")

# to convert original dataset from dat to mat
for gLevel in ["350", "352", "354", "356", "358", "360"]:
   for i in range(1, 216+1):
       srcFolder = "Raw_Data/original/G1_{}/AAA-{}/HypProc_2".format(gLevel, i)
       dstFolder = "Raw_Data/original_mat/{}/{}".format(gLevel, i)
       dp.datToMatBatch(srcFolder=srcFolder, dstFolder=dstFolder)