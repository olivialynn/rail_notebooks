GPz Estimation Example
======================

**Author:** Sam Schmidt

**Last Run Successfully:** September 26, 2023

A quick demo of running GPz on the typical test data. You should have
installed rail_gpz_v1 (we highly recommend that you do this from within
a custom conda environment so that all dependencies for package versions
are met), either by cloning and installing from github, or with:

::

   pip install pz-rail-gpz-v1

As RAIL is a namespace package, installing rail_gpz_v1 will make
``GPzInformer`` and ``GPzEstimator`` available, and they can be imported
via:

::

   from rail.estimation.algos.gpz import GPzInformer, GPzEstimator

Let’s start with all of our necessary imports:

.. code:: ipython3

    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import rail
    import qp
    from rail.core.data import TableHandle
    from rail.core.stage import RailStage
    from rail.estimation.algos.gpz import GPzInformer, GPzEstimator

.. code:: ipython3

    # set up the DataStore to keep track of data
    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True

.. code:: ipython3

    # find_rail_file is a convenience function that finds a file in the RAIL ecosystem   We have several example data files that are copied with RAIL that we can use for our example run, let's grab those files, one for training/validation, and the other for testing:
    from rail.utils.path_utils import find_rail_file
    trainFile = find_rail_file('examples_data/testdata/test_dc2_training_9816.hdf5')
    testFile = find_rail_file('examples_data/testdata/test_dc2_validation_9816.hdf5')
    training_data = DS.read_file("training_data", TableHandle, trainFile)
    test_data = DS.read_file("test_data", TableHandle, testFile)

Now, we need to set up the stage that will run GPz. We begin by defining
a dictionary with the config options for the algorithm. There are
sensible defaults set, we will override several of these as an example
of how to do this. Config parameters not set in the dictionary will
automatically be set to their default values.

.. code:: ipython3

    gpz_train_dict = dict(n_basis=60, trainfrac=0.8, csl_method="normal", max_iter=150, hdf5_groupname="photometry") 

Let’s set up the training stage. We need to provide a name for the stage
for ceci, as well as a name for the model file that will be written by
the stage. We also include the arguments in the dictionary we wrote
above as additional arguments:

.. code:: ipython3

    # set up the stage to run our GPZ_training
    pz_train = GPzInformer.make_stage(name="GPz_Train", model="GPz_model.pkl", **gpz_train_dict)

We are now ready to run the stage to create the model. We will use the
training data from ``test_dc2_training_9816.hdf5``, which contains
10,225 galaxies drawn from healpix 9816 from the cosmoDC2_v1.1.4
dataset, to train the model. Note that we read this data in called
``train_data`` in the DataStore. Note that we set ``trainfrac`` to 0.8,
so 80% of the data will be used in the “main” training, but 20% will be
reserved by ``GPzInformer`` to determine a SIGMA parameter. We set
``max_iter`` to 150, so we will see 150 steps where the stage tries to
maximize the likelihood. We run the stage as follows:

.. code:: ipython3

    %%time
    pz_train.inform(training_data)


.. parsed-literal::

    Inserting handle into data store.  input: None, GPz_Train
    ngal: 10225
    training model...


.. parsed-literal::

    Iter	 logML/n 		 Train RMSE		 Train RMSE/n		 Valid RMSE		 Valid MLL		 Time    
       1	-3.3171704e-01	 3.1695856e-01	-3.2189510e-01	 3.3638784e-01	[-3.5956233e-01]	 4.7078180e-01


.. parsed-literal::

       2	-2.5640306e-01	 3.0430635e-01	-2.3035236e-01	 3.2069198e-01	[-2.8325402e-01]	 2.3568940e-01


.. parsed-literal::

       3	-2.1099938e-01	 2.8407504e-01	-1.6681046e-01	 3.0405771e-01	[-2.4511338e-01]	 2.8948259e-01


.. parsed-literal::

       4	-1.7415360e-01	 2.6779653e-01	-1.2511084e-01	 2.8302950e-01	[-1.9916575e-01]	 2.9538083e-01


.. parsed-literal::

       5	-1.2065332e-01	 2.5593317e-01	-8.8396963e-02	 2.6568118e-01	[-1.4144932e-01]	 2.1015453e-01


.. parsed-literal::

       6	-6.5595616e-02	 2.5131797e-01	-3.7304709e-02	 2.6137939e-01	[-7.6052495e-02]	 2.0867705e-01


.. parsed-literal::

       7	-4.8470834e-02	 2.4768101e-01	-2.4177926e-02	 2.5550272e-01	[-5.4682969e-02]	 2.1010661e-01


.. parsed-literal::

       8	-3.6502566e-02	 2.4584758e-01	-1.5987184e-02	 2.5374837e-01	[-4.8062531e-02]	 2.1288466e-01
       9	-2.2396947e-02	 2.4297511e-01	-5.3447839e-03	 2.5241434e-01	[-4.0482239e-02]	 1.8643117e-01


.. parsed-literal::

      10	-1.1703768e-02	 2.4127940e-01	 3.4473740e-03	 2.4986045e-01	[-3.3948223e-02]	 2.0904446e-01


.. parsed-literal::

      11	-7.2019165e-03	 2.4046838e-01	 7.5386050e-03	 2.4857271e-01	[-2.7373993e-02]	 2.1301460e-01
      12	-2.1149849e-03	 2.3958152e-01	 1.2026241e-02	 2.4714557e-01	[-1.9592365e-02]	 1.6954184e-01


.. parsed-literal::

      13	 2.1758046e-03	 2.3877667e-01	 1.6259151e-02	 2.4584203e-01	[-1.3116341e-02]	 2.1571493e-01


.. parsed-literal::

      14	 1.1553551e-01	 2.2442564e-01	 1.3652005e-01	 2.3214321e-01	[ 1.0956031e-01]	 4.5416832e-01
      15	 1.6790333e-01	 2.2072262e-01	 1.9026385e-01	 2.2862193e-01	[ 1.5919558e-01]	 2.0644879e-01


.. parsed-literal::

      16	 2.6021883e-01	 2.1599103e-01	 2.8831442e-01	 2.2334410e-01	[ 2.5338304e-01]	 1.9066906e-01
      17	 3.2323474e-01	 2.1126960e-01	 3.5515944e-01	 2.1687830e-01	[ 3.1950794e-01]	 1.8723774e-01


.. parsed-literal::

      18	 3.7270420e-01	 2.0928733e-01	 4.0561810e-01	 2.1469411e-01	[ 3.7231659e-01]	 1.9636726e-01


.. parsed-literal::

      19	 4.3491478e-01	 2.0466039e-01	 4.6913100e-01	 2.1170511e-01	[ 4.3102275e-01]	 2.0984173e-01
      20	 5.0977539e-01	 2.0197642e-01	 5.4547289e-01	 2.0781713e-01	[ 5.0913321e-01]	 1.7747068e-01


.. parsed-literal::

      21	 5.9527031e-01	 1.9694107e-01	 6.3343462e-01	 2.0166381e-01	[ 5.8809736e-01]	 2.0524788e-01


.. parsed-literal::

      22	 6.1122726e-01	 1.9432541e-01	 6.5248233e-01	 2.0075555e-01	  5.8019633e-01 	 2.0918965e-01


.. parsed-literal::

      23	 6.5700670e-01	 1.8979163e-01	 6.9527788e-01	 1.9557027e-01	[ 6.3576453e-01]	 2.1245694e-01


.. parsed-literal::

      24	 6.8357235e-01	 1.8708167e-01	 7.2168053e-01	 1.9338596e-01	[ 6.5974807e-01]	 2.1353436e-01


.. parsed-literal::

      25	 7.2277638e-01	 1.8509082e-01	 7.6101709e-01	 1.9246863e-01	[ 7.0004781e-01]	 2.0586514e-01
      26	 7.4731563e-01	 1.9204906e-01	 7.8556982e-01	 2.0104880e-01	[ 7.2822258e-01]	 1.6730881e-01


.. parsed-literal::

      27	 7.7771136e-01	 1.8839714e-01	 8.1625264e-01	 1.9838557e-01	[ 7.5568119e-01]	 2.1108580e-01


.. parsed-literal::

      28	 7.9585894e-01	 1.8397136e-01	 8.3440068e-01	 1.9342356e-01	[ 7.7795500e-01]	 2.0320177e-01


.. parsed-literal::

      29	 8.2532467e-01	 1.8019304e-01	 8.6459294e-01	 1.8888925e-01	[ 8.1899142e-01]	 2.0735979e-01


.. parsed-literal::

      30	 8.5395686e-01	 1.7717992e-01	 8.9369641e-01	 1.8556984e-01	[ 8.5462085e-01]	 2.1233559e-01
      31	 8.7951177e-01	 1.7586413e-01	 9.2020314e-01	 1.8590402e-01	[ 8.8650134e-01]	 1.7208052e-01


.. parsed-literal::

      32	 8.9988392e-01	 1.7190821e-01	 9.4072752e-01	 1.8114225e-01	[ 9.1544927e-01]	 2.0457840e-01
      33	 9.1652312e-01	 1.6934585e-01	 9.5744094e-01	 1.7832727e-01	[ 9.3316434e-01]	 1.9772601e-01


.. parsed-literal::

      34	 9.3878397e-01	 1.6763117e-01	 9.8044866e-01	 1.7806866e-01	[ 9.5910012e-01]	 1.8798018e-01
      35	 9.6381099e-01	 1.6471600e-01	 1.0068203e+00	 1.7574551e-01	[ 9.8533878e-01]	 1.8963313e-01


.. parsed-literal::

      36	 9.7667493e-01	 1.6075965e-01	 1.0208520e+00	 1.7226315e-01	[ 1.0019419e+00]	 2.0901346e-01


.. parsed-literal::

      37	 9.9240418e-01	 1.5942973e-01	 1.0361541e+00	 1.7078081e-01	[ 1.0156132e+00]	 2.1995950e-01


.. parsed-literal::

      38	 1.0057019e+00	 1.5851208e-01	 1.0494524e+00	 1.7003652e-01	[ 1.0270699e+00]	 2.1501303e-01


.. parsed-literal::

      39	 1.0210003e+00	 1.5714237e-01	 1.0650382e+00	 1.6893732e-01	[ 1.0423813e+00]	 2.1204090e-01
      40	 1.0437467e+00	 1.5468301e-01	 1.0889976e+00	 1.6817427e-01	[ 1.0592702e+00]	 1.9472289e-01


.. parsed-literal::

      41	 1.0524045e+00	 1.5413736e-01	 1.0977752e+00	 1.6781124e-01	[ 1.0690557e+00]	 3.0443215e-01
      42	 1.0612026e+00	 1.5264517e-01	 1.1065990e+00	 1.6654499e-01	[ 1.0756514e+00]	 2.0520973e-01


.. parsed-literal::

      43	 1.0774873e+00	 1.4920276e-01	 1.1233049e+00	 1.6370262e-01	[ 1.0853060e+00]	 2.0733953e-01


.. parsed-literal::

      44	 1.0860605e+00	 1.4809436e-01	 1.1319640e+00	 1.6253782e-01	[ 1.0957131e+00]	 2.1368885e-01


.. parsed-literal::

      45	 1.0951790e+00	 1.4655663e-01	 1.1412226e+00	 1.6122593e-01	[ 1.1060555e+00]	 2.1152902e-01


.. parsed-literal::

      46	 1.1050625e+00	 1.4530341e-01	 1.1511387e+00	 1.6037030e-01	[ 1.1173050e+00]	 2.1155930e-01


.. parsed-literal::

      47	 1.1162821e+00	 1.4414868e-01	 1.1628086e+00	 1.6003361e-01	[ 1.1278882e+00]	 2.1009612e-01
      48	 1.1286746e+00	 1.4324510e-01	 1.1755361e+00	 1.6123292e-01	[ 1.1370540e+00]	 1.9716239e-01


.. parsed-literal::

      49	 1.1403139e+00	 1.4228737e-01	 1.1872560e+00	 1.6148973e-01	[ 1.1431151e+00]	 2.0955777e-01
      50	 1.1493973e+00	 1.4131562e-01	 1.1967002e+00	 1.6170802e-01	[ 1.1471661e+00]	 1.9955158e-01


.. parsed-literal::

      51	 1.1578399e+00	 1.4068301e-01	 1.2053239e+00	 1.6121847e-01	[ 1.1511095e+00]	 2.0645022e-01
      52	 1.1672596e+00	 1.3983815e-01	 1.2149231e+00	 1.5971208e-01	[ 1.1622723e+00]	 1.9837475e-01


.. parsed-literal::

      53	 1.1760103e+00	 1.3867226e-01	 1.2241273e+00	 1.5861402e-01	[ 1.1722023e+00]	 2.1081972e-01


.. parsed-literal::

      54	 1.1834850e+00	 1.3866642e-01	 1.2314863e+00	 1.5788965e-01	[ 1.1826439e+00]	 2.1511960e-01


.. parsed-literal::

      55	 1.1885411e+00	 1.3812665e-01	 1.2365181e+00	 1.5749314e-01	[ 1.1882247e+00]	 2.1290326e-01


.. parsed-literal::

      56	 1.1980620e+00	 1.3747246e-01	 1.2462706e+00	 1.5739948e-01	[ 1.1974985e+00]	 2.1351981e-01


.. parsed-literal::

      57	 1.2041361e+00	 1.3727661e-01	 1.2524213e+00	 1.5664505e-01	[ 1.2024101e+00]	 2.0980239e-01


.. parsed-literal::

      58	 1.2136995e+00	 1.3668982e-01	 1.2618641e+00	 1.5639277e-01	[ 1.2127701e+00]	 2.1060610e-01
      59	 1.2229562e+00	 1.3583833e-01	 1.2712298e+00	 1.5528472e-01	[ 1.2223043e+00]	 1.8440342e-01


.. parsed-literal::

      60	 1.2314844e+00	 1.3525112e-01	 1.2799760e+00	 1.5386092e-01	[ 1.2290260e+00]	 1.9116545e-01


.. parsed-literal::

      61	 1.2372249e+00	 1.3477160e-01	 1.2859987e+00	 1.5224301e-01	[ 1.2344446e+00]	 3.3056688e-01


.. parsed-literal::

      62	 1.2445442e+00	 1.3438209e-01	 1.2935636e+00	 1.5103829e-01	[ 1.2396593e+00]	 2.1559429e-01


.. parsed-literal::

      63	 1.2518590e+00	 1.3401483e-01	 1.3011574e+00	 1.5040060e-01	[ 1.2430914e+00]	 2.1146917e-01


.. parsed-literal::

      64	 1.2591100e+00	 1.3349502e-01	 1.3089081e+00	 1.4992680e-01	[ 1.2444677e+00]	 2.0512414e-01


.. parsed-literal::

      65	 1.2658950e+00	 1.3347659e-01	 1.3158793e+00	 1.5016406e-01	[ 1.2449727e+00]	 2.0301795e-01
      66	 1.2717651e+00	 1.3310388e-01	 1.3216728e+00	 1.5030461e-01	[ 1.2484679e+00]	 1.9854212e-01


.. parsed-literal::

      67	 1.2816045e+00	 1.3227996e-01	 1.3320656e+00	 1.5065638e-01	[ 1.2511911e+00]	 1.9396019e-01


.. parsed-literal::

      68	 1.2860928e+00	 1.3223224e-01	 1.3368568e+00	 1.5056595e-01	  1.2472681e+00 	 2.1094418e-01


.. parsed-literal::

      69	 1.2923349e+00	 1.3201463e-01	 1.3430197e+00	 1.5046650e-01	[ 1.2539047e+00]	 2.0229602e-01
      70	 1.2968447e+00	 1.3197942e-01	 1.3475331e+00	 1.5017016e-01	[ 1.2578649e+00]	 2.0093560e-01


.. parsed-literal::

      71	 1.3021238e+00	 1.3189411e-01	 1.3527795e+00	 1.4989270e-01	[ 1.2635047e+00]	 1.9337225e-01


.. parsed-literal::

      72	 1.3105905e+00	 1.3181238e-01	 1.3614735e+00	 1.4886757e-01	[ 1.2764335e+00]	 2.2083807e-01
      73	 1.3168949e+00	 1.3148161e-01	 1.3678511e+00	 1.4867824e-01	[ 1.2876521e+00]	 1.8272662e-01


.. parsed-literal::

      74	 1.3215965e+00	 1.3105108e-01	 1.3724918e+00	 1.4842188e-01	[ 1.2912128e+00]	 1.8560839e-01


.. parsed-literal::

      75	 1.3264713e+00	 1.3028275e-01	 1.3774350e+00	 1.4810264e-01	[ 1.2957009e+00]	 2.1186614e-01
      76	 1.3321610e+00	 1.2942925e-01	 1.3834065e+00	 1.4749347e-01	[ 1.3010127e+00]	 1.9781041e-01


.. parsed-literal::

      77	 1.3384997e+00	 1.2852841e-01	 1.3901570e+00	 1.4707875e-01	[ 1.3042408e+00]	 2.1242976e-01


.. parsed-literal::

      78	 1.3439599e+00	 1.2831012e-01	 1.3955616e+00	 1.4660585e-01	[ 1.3086694e+00]	 2.1897244e-01


.. parsed-literal::

      79	 1.3497965e+00	 1.2805880e-01	 1.4013809e+00	 1.4596985e-01	[ 1.3114287e+00]	 2.1038747e-01


.. parsed-literal::

      80	 1.3558117e+00	 1.2748742e-01	 1.4073295e+00	 1.4532700e-01	[ 1.3148488e+00]	 2.2070980e-01


.. parsed-literal::

      81	 1.3614006e+00	 1.2656479e-01	 1.4130633e+00	 1.4433297e-01	  1.3090664e+00 	 2.1106267e-01


.. parsed-literal::

      82	 1.3672725e+00	 1.2611004e-01	 1.4188267e+00	 1.4392217e-01	[ 1.3157478e+00]	 2.1228766e-01


.. parsed-literal::

      83	 1.3706929e+00	 1.2568363e-01	 1.4222612e+00	 1.4368829e-01	[ 1.3188166e+00]	 2.2508550e-01


.. parsed-literal::

      84	 1.3762852e+00	 1.2468907e-01	 1.4281865e+00	 1.4308482e-01	  1.3163958e+00 	 2.1592093e-01


.. parsed-literal::

      85	 1.3811084e+00	 1.2390130e-01	 1.4333646e+00	 1.4180559e-01	  1.3094642e+00 	 2.0334482e-01


.. parsed-literal::

      86	 1.3860985e+00	 1.2365631e-01	 1.4383158e+00	 1.4149386e-01	  1.3119206e+00 	 2.0089507e-01


.. parsed-literal::

      87	 1.3898861e+00	 1.2342597e-01	 1.4421407e+00	 1.4105083e-01	  1.3128058e+00 	 2.1950841e-01


.. parsed-literal::

      88	 1.3950089e+00	 1.2303938e-01	 1.4473559e+00	 1.4071866e-01	  1.3146931e+00 	 2.1409106e-01
      89	 1.3988400e+00	 1.2244504e-01	 1.4517189e+00	 1.3964709e-01	  1.3174749e+00 	 1.9263601e-01


.. parsed-literal::

      90	 1.4036531e+00	 1.2223214e-01	 1.4564438e+00	 1.3968568e-01	[ 1.3216221e+00]	 2.1588826e-01
      91	 1.4061257e+00	 1.2210132e-01	 1.4588256e+00	 1.3983940e-01	[ 1.3255185e+00]	 1.7966390e-01


.. parsed-literal::

      92	 1.4103719e+00	 1.2180654e-01	 1.4632267e+00	 1.3984712e-01	  1.3243182e+00 	 2.0448565e-01
      93	 1.4160334e+00	 1.2142054e-01	 1.4690924e+00	 1.3981430e-01	  1.3175838e+00 	 1.9508910e-01


.. parsed-literal::

      94	 1.4202458e+00	 1.2139434e-01	 1.4732222e+00	 1.3991669e-01	  1.3103724e+00 	 1.9680882e-01


.. parsed-literal::

      95	 1.4231698e+00	 1.2130255e-01	 1.4759471e+00	 1.3963311e-01	  1.3155093e+00 	 2.1393204e-01


.. parsed-literal::

      96	 1.4260109e+00	 1.2112807e-01	 1.4787067e+00	 1.3938660e-01	  1.3171261e+00 	 2.0852947e-01
      97	 1.4297823e+00	 1.2093723e-01	 1.4825388e+00	 1.3932346e-01	  1.3144406e+00 	 1.9670677e-01


.. parsed-literal::

      98	 1.4319000e+00	 1.2071157e-01	 1.4848645e+00	 1.3944186e-01	  1.3128777e+00 	 1.9972992e-01


.. parsed-literal::

      99	 1.4359842e+00	 1.2065295e-01	 1.4889496e+00	 1.3962652e-01	  1.3102780e+00 	 2.0773387e-01


.. parsed-literal::

     100	 1.4378736e+00	 1.2058339e-01	 1.4908681e+00	 1.3973138e-01	  1.3103757e+00 	 2.1558475e-01


.. parsed-literal::

     101	 1.4405968e+00	 1.2038988e-01	 1.4936902e+00	 1.3997719e-01	  1.3080098e+00 	 2.1192575e-01


.. parsed-literal::

     102	 1.4424327e+00	 1.2003731e-01	 1.4956953e+00	 1.3980855e-01	  1.3056288e+00 	 2.1710134e-01
     103	 1.4462094e+00	 1.1986325e-01	 1.4993806e+00	 1.3990635e-01	  1.3079483e+00 	 1.9961691e-01


.. parsed-literal::

     104	 1.4482137e+00	 1.1971273e-01	 1.5013315e+00	 1.3979701e-01	  1.3104443e+00 	 1.7945600e-01


.. parsed-literal::

     105	 1.4502401e+00	 1.1956305e-01	 1.5033695e+00	 1.3972091e-01	  1.3123071e+00 	 2.1708131e-01


.. parsed-literal::

     106	 1.4544587e+00	 1.1930222e-01	 1.5077027e+00	 1.3992564e-01	  1.3128850e+00 	 2.1940684e-01
     107	 1.4572255e+00	 1.1868756e-01	 1.5107856e+00	 1.4078030e-01	  1.3094825e+00 	 1.9781280e-01


.. parsed-literal::

     108	 1.4620154e+00	 1.1864875e-01	 1.5154958e+00	 1.4128186e-01	  1.3094283e+00 	 1.8724608e-01


.. parsed-literal::

     109	 1.4636081e+00	 1.1853937e-01	 1.5170377e+00	 1.4129982e-01	  1.3085444e+00 	 2.0394516e-01


.. parsed-literal::

     110	 1.4658706e+00	 1.1837326e-01	 1.5193537e+00	 1.4173301e-01	  1.3047437e+00 	 2.1308589e-01
     111	 1.4680131e+00	 1.1817294e-01	 1.5215826e+00	 1.4220447e-01	  1.2988398e+00 	 1.8043995e-01


.. parsed-literal::

     112	 1.4702150e+00	 1.1805861e-01	 1.5237843e+00	 1.4222693e-01	  1.3011442e+00 	 2.1386766e-01


.. parsed-literal::

     113	 1.4726502e+00	 1.1802262e-01	 1.5262709e+00	 1.4234350e-01	  1.3022921e+00 	 2.0156884e-01
     114	 1.4743559e+00	 1.1804058e-01	 1.5279967e+00	 1.4236139e-01	  1.3028839e+00 	 1.8811774e-01


.. parsed-literal::

     115	 1.4778494e+00	 1.1814913e-01	 1.5316706e+00	 1.4267683e-01	  1.2973872e+00 	 2.1168995e-01


.. parsed-literal::

     116	 1.4806390e+00	 1.1818580e-01	 1.5344991e+00	 1.4253123e-01	  1.2980184e+00 	 2.1321750e-01


.. parsed-literal::

     117	 1.4823625e+00	 1.1803200e-01	 1.5361576e+00	 1.4231103e-01	  1.2987318e+00 	 2.2147417e-01
     118	 1.4846842e+00	 1.1781035e-01	 1.5385634e+00	 1.4244536e-01	  1.2946884e+00 	 1.9133806e-01


.. parsed-literal::

     119	 1.4868509e+00	 1.1767211e-01	 1.5408895e+00	 1.4276322e-01	  1.2931039e+00 	 2.0939875e-01


.. parsed-literal::

     120	 1.4898176e+00	 1.1755029e-01	 1.5440262e+00	 1.4312363e-01	  1.2867306e+00 	 2.0829391e-01


.. parsed-literal::

     121	 1.4925267e+00	 1.1756495e-01	 1.5468412e+00	 1.4380380e-01	  1.2811614e+00 	 2.1311569e-01


.. parsed-literal::

     122	 1.4949287e+00	 1.1761743e-01	 1.5491459e+00	 1.4399315e-01	  1.2775950e+00 	 2.1976209e-01


.. parsed-literal::

     123	 1.4966613e+00	 1.1765775e-01	 1.5507833e+00	 1.4399094e-01	  1.2771811e+00 	 2.1451783e-01
     124	 1.4988374e+00	 1.1756119e-01	 1.5529135e+00	 1.4403337e-01	  1.2697193e+00 	 1.9343758e-01


.. parsed-literal::

     125	 1.5003911e+00	 1.1731940e-01	 1.5546092e+00	 1.4475567e-01	  1.2665876e+00 	 2.0656610e-01


.. parsed-literal::

     126	 1.5030724e+00	 1.1719950e-01	 1.5572848e+00	 1.4460963e-01	  1.2599327e+00 	 2.1780729e-01


.. parsed-literal::

     127	 1.5043049e+00	 1.1708793e-01	 1.5585584e+00	 1.4452002e-01	  1.2578834e+00 	 2.1307802e-01


.. parsed-literal::

     128	 1.5060380e+00	 1.1694814e-01	 1.5603936e+00	 1.4450426e-01	  1.2533761e+00 	 2.0162702e-01


.. parsed-literal::

     129	 1.5081741e+00	 1.1689814e-01	 1.5626136e+00	 1.4424368e-01	  1.2491654e+00 	 2.2335386e-01


.. parsed-literal::

     130	 1.5106447e+00	 1.1672298e-01	 1.5651045e+00	 1.4404490e-01	  1.2458389e+00 	 2.1019864e-01


.. parsed-literal::

     131	 1.5123595e+00	 1.1665182e-01	 1.5667466e+00	 1.4379336e-01	  1.2449389e+00 	 2.0752883e-01
     132	 1.5140665e+00	 1.1659748e-01	 1.5683943e+00	 1.4363428e-01	  1.2446144e+00 	 2.0186639e-01


.. parsed-literal::

     133	 1.5162181e+00	 1.1654020e-01	 1.5705671e+00	 1.4344656e-01	  1.2344719e+00 	 2.0485592e-01


.. parsed-literal::

     134	 1.5182446e+00	 1.1654694e-01	 1.5726131e+00	 1.4352860e-01	  1.2353462e+00 	 2.0130515e-01


.. parsed-literal::

     135	 1.5197008e+00	 1.1649856e-01	 1.5740911e+00	 1.4355362e-01	  1.2364374e+00 	 2.1040511e-01


.. parsed-literal::

     136	 1.5212250e+00	 1.1639943e-01	 1.5757152e+00	 1.4354197e-01	  1.2332326e+00 	 2.1500087e-01


.. parsed-literal::

     137	 1.5232302e+00	 1.1633835e-01	 1.5778460e+00	 1.4362930e-01	  1.2244541e+00 	 2.0646405e-01


.. parsed-literal::

     138	 1.5253258e+00	 1.1629669e-01	 1.5802526e+00	 1.4377905e-01	  1.2074689e+00 	 2.1093464e-01


.. parsed-literal::

     139	 1.5271873e+00	 1.1632799e-01	 1.5820163e+00	 1.4392801e-01	  1.1989378e+00 	 2.1156836e-01
     140	 1.5280777e+00	 1.1634186e-01	 1.5828029e+00	 1.4400276e-01	  1.2038159e+00 	 1.8757868e-01


.. parsed-literal::

     141	 1.5294422e+00	 1.1632512e-01	 1.5840968e+00	 1.4418295e-01	  1.2053275e+00 	 2.0997834e-01


.. parsed-literal::

     142	 1.5312804e+00	 1.1626835e-01	 1.5859580e+00	 1.4469523e-01	  1.2038739e+00 	 2.0923829e-01
     143	 1.5328127e+00	 1.1625321e-01	 1.5875383e+00	 1.4489348e-01	  1.1961724e+00 	 1.9827390e-01


.. parsed-literal::

     144	 1.5339895e+00	 1.1621439e-01	 1.5887671e+00	 1.4489286e-01	  1.1940223e+00 	 2.1095777e-01


.. parsed-literal::

     145	 1.5354764e+00	 1.1622494e-01	 1.5904328e+00	 1.4486081e-01	  1.1812418e+00 	 2.1991086e-01


.. parsed-literal::

     146	 1.5363949e+00	 1.1624195e-01	 1.5914018e+00	 1.4459916e-01	  1.1828318e+00 	 2.0899343e-01
     147	 1.5373998e+00	 1.1625097e-01	 1.5923939e+00	 1.4447646e-01	  1.1822475e+00 	 1.9038653e-01


.. parsed-literal::

     148	 1.5392085e+00	 1.1628335e-01	 1.5942277e+00	 1.4421033e-01	  1.1797511e+00 	 2.0785642e-01


.. parsed-literal::

     149	 1.5401512e+00	 1.1625032e-01	 1.5951918e+00	 1.4394949e-01	  1.1827041e+00 	 2.1003318e-01


.. parsed-literal::

     150	 1.5413454e+00	 1.1620945e-01	 1.5963546e+00	 1.4381656e-01	  1.1856294e+00 	 2.2084117e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 6s, sys: 1.13 s, total: 2min 7s
    Wall time: 32 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fb5dc2a0f40>



This should have taken about 30 seconds on a typical desktop computer,
and you should now see a file called ``GPz_model.pkl`` in the directory.
This model file is used by the ``GPzEstimator`` stage to determine our
redshift PDFs for the test set of galaxies. Let’s set up that stage,
again defining a dictionary of variables for the config params:

.. code:: ipython3

    gpz_test_dict = dict(hdf5_groupname="photometry", model="GPz_model.pkl")
    
    gpz_run = GPzEstimator.make_stage(name="gpz_run", **gpz_test_dict)

Let’s run the stage and compute photo-z’s for our test set:

.. code:: ipython3

    %%time
    results = gpz_run.estimate(test_data)


.. parsed-literal::

    Inserting handle into data store.  model: GPz_model.pkl, gpz_run
    Process 0 running estimator on chunk 0 - 10,000
    Process 0 estimating GPz PZ PDF for rows 0 - 10,000


.. parsed-literal::

    Inserting handle into data store.  output_gpz_run: inprogress_output_gpz_run.hdf5, gpz_run


.. parsed-literal::

    Process 0 running estimator on chunk 10,000 - 20,000
    Process 0 estimating GPz PZ PDF for rows 10,000 - 20,000


.. parsed-literal::

    Process 0 running estimator on chunk 20,000 - 20,449
    Process 0 estimating GPz PZ PDF for rows 20,000 - 20,449
    CPU times: user 1.88 s, sys: 40 ms, total: 1.92 s
    Wall time: 631 ms


This should be very fast, under a second for our 20,449 galaxies in the
test set. Now, let’s plot a scatter plot of the point estimates, as well
as a few example PDFs. We can get access to the ``qp`` ensemble that was
written via the DataStore via ``results()``

.. code:: ipython3

    ens = results()

.. code:: ipython3

    expdfids = [2, 180, 13517, 18032]
    fig, axs = plt.subplots(4, 1, figsize=(12,10))
    for i, xx in enumerate(expdfids):
        axs[i].set_xlim(0,3)
        ens[xx].plot_native(axes=axs[i])
    axs[3].set_xlabel("redshift", fontsize=15)




.. parsed-literal::

    Text(0.5, 0, 'redshift')




.. image:: ../../../docs/rendered/estimation_examples/06_GPz_files/../../../docs/rendered/estimation_examples/06_GPz_16_1.png


GPzEstimator parameterizes each PDF as a single Gaussian, here we see a
few examples of Gaussians of different widths. Now let’s grab the mode
of each PDF (stored as ancil data in the ensemble) and compare to the
true redshifts from the test_data file:

.. code:: ipython3

    truez = test_data.data['photometry']['redshift']
    zmode = ens.ancil['zmode'].flatten()

.. code:: ipython3

    plt.figure(figsize=(12,12))
    plt.scatter(truez, zmode, s=3)
    plt.plot([0,3],[0,3], 'k--')
    plt.xlabel("redshift", fontsize=12)
    plt.ylabel("z mode", fontsize=12)




.. parsed-literal::

    Text(0, 0.5, 'z mode')




.. image:: ../../../docs/rendered/estimation_examples/06_GPz_files/../../../docs/rendered/estimation_examples/06_GPz_19_1.png

