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

    # RAILDIR is a convenience variable set within RAIL that stores the path to the package as installed on your machine.  We have several example data files that are copied with RAIL that we can use for our example run, let's grab those files, one for training/validation, and the other for testing:
    from rail.core.utils import RAILDIR
    trainFile = os.path.join(RAILDIR, 'rail/examples_data/testdata/test_dc2_training_9816.hdf5')
    testFile = os.path.join(RAILDIR, 'rail/examples_data/testdata/test_dc2_validation_9816.hdf5')
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

    ngal: 10225
    training model...
    Iter	 logML/n 		 Train RMSE		 Train RMSE/n		 Valid RMSE		 Valid MLL		 Time    
       1	-3.0464765e-01	 3.2309715e-01	-2.9037067e-01	 2.9701013e-01	[-2.1966205e-01]	 8.0568671e-01
       2	-2.3401113e-01	 2.9191640e-01	-1.9748663e-01	 2.7626494e-01	[-1.3409275e-01]	 1.8716908e-01
       3	-1.6797840e-01	 2.6738860e-01	-1.4177046e-01	 2.5976835e-01	[-9.8854669e-02]	 1.8908310e-01
       4	-8.7839309e-02	 2.5800838e-01	-6.5295087e-02	 2.4687634e-01	[-2.5903521e-02]	 2.0806956e-01
       5	-6.0728709e-02	 2.5214570e-01	-4.1243426e-02	 2.3958477e-01	[ 8.7003579e-03]	 1.9330454e-01
       6	-4.4124119e-02	 2.4841361e-01	-2.5908680e-02	 2.3685641e-01	[ 1.9460801e-02]	 2.0404220e-01
       7	-3.3784644e-02	 2.4614122e-01	-1.6557317e-02	 2.3535460e-01	[ 2.6691456e-02]	 1.9453764e-01
       8	-2.4168063e-02	 2.4418213e-01	-8.1959454e-03	 2.3391208e-01	[ 3.2332696e-02]	 1.8979144e-01
       9	-1.9333350e-02	 2.4321749e-01	-4.0152884e-03	 2.3299440e-01	[ 3.7226802e-02]	 1.9338322e-01
      10	-1.5850225e-02	 2.4249447e-01	-9.3359636e-04	 2.3290010e-01	  3.6839507e-02 	 1.8729615e-01
      11	-1.1667955e-02	 2.4159165e-01	 3.1184917e-03	 2.3269210e-01	[ 3.8148855e-02]	 1.8189049e-01
      12	-3.9115594e-03	 2.3990574e-01	 1.1492450e-02	 2.3263423e-01	[ 3.9897299e-02]	 1.9300365e-01
      13	 2.1094409e-02	 2.3479430e-01	 3.8471841e-02	 2.3020891e-01	[ 5.7447638e-02]	 1.9670534e-01
      14	 7.6433196e-02	 2.3115449e-01	 1.0087041e-01	 2.2617141e-01	[ 1.1824218e-01]	 1.8697786e-01
      15	 7.8405062e-02	 2.2984943e-01	 1.0634579e-01	 2.2773965e-01	  1.1117686e-01 	 1.9598627e-01
      16	 2.2144676e-01	 2.2126192e-01	 2.5201307e-01	 2.2332313e-01	[ 2.4804122e-01]	 2.0748997e-01
      17	 2.7851542e-01	 2.1786752e-01	 3.1148821e-01	 2.1877899e-01	[ 3.1181777e-01]	 3.8162827e-01
      18	 3.0967244e-01	 2.1443757e-01	 3.4382567e-01	 2.1662363e-01	[ 3.3399111e-01]	 1.9358611e-01
      19	 3.4307554e-01	 2.1244977e-01	 3.7771764e-01	 2.1587455e-01	[ 3.6200470e-01]	 2.0189977e-01
      20	 4.0282715e-01	 2.0962579e-01	 4.3941401e-01	 2.1387867e-01	[ 4.1818012e-01]	 2.0485687e-01
      21	 4.4771140e-01	 2.0832997e-01	 4.8658171e-01	 2.0976595e-01	[ 4.6511189e-01]	 1.9950628e-01
      22	 4.8782122e-01	 2.0350433e-01	 5.2693368e-01	 2.0638432e-01	[ 5.0218013e-01]	 2.0825291e-01
      23	 5.3453091e-01	 2.0060103e-01	 5.7539533e-01	 2.0493576e-01	[ 5.4326355e-01]	 2.0429134e-01
      24	 5.7431942e-01	 1.9930169e-01	 6.1308791e-01	 2.0234977e-01	[ 5.9817239e-01]	 1.8996668e-01
      25	 6.1269910e-01	 1.9682277e-01	 6.5303887e-01	 1.9887255e-01	[ 6.3702251e-01]	 1.9095325e-01
      26	 6.3902873e-01	 1.9855755e-01	 6.8142098e-01	 1.9775049e-01	[ 6.6601700e-01]	 1.9602799e-01
      27	 6.8481386e-01	 1.9803337e-01	 7.2568936e-01	 1.9785188e-01	[ 7.1221236e-01]	 2.0395088e-01
      28	 7.3104077e-01	 2.0333548e-01	 7.7173885e-01	 2.0212690e-01	[ 7.6686474e-01]	 1.7997432e-01
      29	 7.6481228e-01	 2.0592962e-01	 8.0586477e-01	 2.0365846e-01	[ 8.0494397e-01]	 1.9369578e-01
      30	 7.9720000e-01	 2.0491006e-01	 8.3857232e-01	 2.0250359e-01	[ 8.4363376e-01]	 1.8108654e-01
      31	 8.2635404e-01	 2.0282494e-01	 8.6913344e-01	 1.9934288e-01	[ 8.7574462e-01]	 1.8818641e-01
      32	 8.5316668e-01	 1.9953327e-01	 8.9820849e-01	 1.9497337e-01	[ 8.8804687e-01]	 1.7845249e-01
      33	 8.7271342e-01	 1.9810785e-01	 9.1732700e-01	 1.9283844e-01	[ 9.1325982e-01]	 1.8294549e-01
      34	 8.8358587e-01	 1.9771034e-01	 9.2859222e-01	 1.9318220e-01	[ 9.2382424e-01]	 1.9489121e-01
      35	 8.9990417e-01	 1.9537400e-01	 9.4429283e-01	 1.9121870e-01	[ 9.3592109e-01]	 1.8235588e-01
      36	 9.1672445e-01	 1.9376539e-01	 9.6138224e-01	 1.8993631e-01	[ 9.4693279e-01]	 1.8631792e-01
      37	 9.4065222e-01	 1.9035279e-01	 9.8649589e-01	 1.8775378e-01	[ 9.5942174e-01]	 1.9204831e-01
      38	 9.5723399e-01	 1.8909530e-01	 1.0036455e+00	 1.8504037e-01	[ 9.8209632e-01]	 1.8500519e-01
      39	 9.6979512e-01	 1.8771649e-01	 1.0162475e+00	 1.8460106e-01	[ 9.9389868e-01]	 2.0082974e-01
      40	 9.8232503e-01	 1.8625889e-01	 1.0295588e+00	 1.8329048e-01	[ 1.0056894e+00]	 1.9215631e-01
      41	 9.9479642e-01	 1.8520466e-01	 1.0424456e+00	 1.8134532e-01	[ 1.0172011e+00]	 1.8760943e-01
      42	 1.0028986e+00	 1.8434653e-01	 1.0518151e+00	 1.7776770e-01	[ 1.0299322e+00]	 1.9554138e-01
      43	 1.0145921e+00	 1.8333249e-01	 1.0629036e+00	 1.7695966e-01	[ 1.0381976e+00]	 1.8434882e-01
      44	 1.0202331e+00	 1.8205565e-01	 1.0683270e+00	 1.7677040e-01	[ 1.0410528e+00]	 1.7662334e-01
      45	 1.0337672e+00	 1.7987462e-01	 1.0818447e+00	 1.7582341e-01	[ 1.0485214e+00]	 2.0206952e-01
      46	 1.0470572e+00	 1.7810816e-01	 1.0952543e+00	 1.7406525e-01	[ 1.0544131e+00]	 1.8408680e-01
      47	 1.0481599e+00	 1.7810576e-01	 1.0966404e+00	 1.7060640e-01	  1.0534795e+00 	 1.8536949e-01
      48	 1.0672821e+00	 1.7454975e-01	 1.1155570e+00	 1.6952117e-01	[ 1.0665282e+00]	 1.9299793e-01
      49	 1.0719616e+00	 1.7369884e-01	 1.1202243e+00	 1.6925931e-01	[ 1.0707278e+00]	 1.8534517e-01
      50	 1.0792349e+00	 1.7213866e-01	 1.1276880e+00	 1.6756612e-01	[ 1.0751855e+00]	 1.7767119e-01
      51	 1.0874667e+00	 1.6994639e-01	 1.1363406e+00	 1.6535957e-01	[ 1.0862945e+00]	 1.9196510e-01
      52	 1.0952367e+00	 1.6828272e-01	 1.1442186e+00	 1.6387965e-01	[ 1.0924577e+00]	 1.9638538e-01
      53	 1.1022622e+00	 1.6654352e-01	 1.1513166e+00	 1.6288498e-01	[ 1.0960183e+00]	 1.8756962e-01
      54	 1.1122665e+00	 1.6400044e-01	 1.1614600e+00	 1.6134687e-01	[ 1.1002316e+00]	 1.8419361e-01
      55	 1.1136770e+00	 1.6070649e-01	 1.1636006e+00	 1.5906508e-01	  1.0844960e+00 	 2.0561934e-01
      56	 1.1260202e+00	 1.6046332e-01	 1.1754405e+00	 1.5896714e-01	[ 1.1063542e+00]	 1.8722057e-01
      57	 1.1305588e+00	 1.5996358e-01	 1.1799503e+00	 1.5818959e-01	[ 1.1156601e+00]	 1.8181944e-01
      58	 1.1374124e+00	 1.5936120e-01	 1.1868324e+00	 1.5682909e-01	[ 1.1265655e+00]	 1.8850183e-01
      59	 1.1456223e+00	 1.5827702e-01	 1.1953945e+00	 1.5446085e-01	[ 1.1402965e+00]	 1.7848396e-01
      60	 1.1539130e+00	 1.5783480e-01	 1.2040326e+00	 1.5272853e-01	[ 1.1445892e+00]	 1.7641497e-01
      61	 1.1599462e+00	 1.5727068e-01	 1.2100115e+00	 1.5198997e-01	[ 1.1507721e+00]	 1.9170451e-01
      62	 1.1697030e+00	 1.5599648e-01	 1.2198509e+00	 1.5021410e-01	[ 1.1626309e+00]	 1.8426347e-01
      63	 1.1779974e+00	 1.5600309e-01	 1.2281061e+00	 1.4956735e-01	[ 1.1727786e+00]	 1.8805432e-01
      64	 1.1858744e+00	 1.5520789e-01	 1.2360777e+00	 1.4846568e-01	[ 1.1834432e+00]	 1.7360163e-01
      65	 1.1927716e+00	 1.5416336e-01	 1.2431036e+00	 1.4763731e-01	[ 1.1896509e+00]	 1.8209100e-01
      66	 1.2021959e+00	 1.5265102e-01	 1.2528801e+00	 1.4697606e-01	[ 1.1947649e+00]	 1.7667913e-01
      67	 1.2071601e+00	 1.5109376e-01	 1.2584624e+00	 1.4606789e-01	  1.1914924e+00 	 1.8473911e-01
      68	 1.2147767e+00	 1.5043317e-01	 1.2658006e+00	 1.4630777e-01	[ 1.1995690e+00]	 1.8618631e-01
      69	 1.2205259e+00	 1.4973630e-01	 1.2715929e+00	 1.4591850e-01	[ 1.2041708e+00]	 1.8852663e-01
      70	 1.2262496e+00	 1.4898986e-01	 1.2774713e+00	 1.4564851e-01	[ 1.2078833e+00]	 1.7863369e-01
      71	 1.2303484e+00	 1.4813309e-01	 1.2820563e+00	 1.4478742e-01	[ 1.2108715e+00]	 1.8833280e-01
      72	 1.2375761e+00	 1.4751606e-01	 1.2892160e+00	 1.4406738e-01	[ 1.2177288e+00]	 1.8944335e-01
      73	 1.2412548e+00	 1.4712638e-01	 1.2928829e+00	 1.4346181e-01	[ 1.2218651e+00]	 1.8947458e-01
      74	 1.2467663e+00	 1.4634517e-01	 1.2984778e+00	 1.4233766e-01	[ 1.2278216e+00]	 2.0014191e-01
      75	 1.2512401e+00	 1.4584042e-01	 1.3032113e+00	 1.4097244e-01	[ 1.2313137e+00]	 1.8956065e-01
      76	 1.2585310e+00	 1.4505208e-01	 1.3103720e+00	 1.4023606e-01	[ 1.2394297e+00]	 1.9166374e-01
      77	 1.2623492e+00	 1.4465355e-01	 1.3141412e+00	 1.4013889e-01	[ 1.2432509e+00]	 1.9476390e-01
      78	 1.2677395e+00	 1.4410905e-01	 1.3194966e+00	 1.3992828e-01	[ 1.2461854e+00]	 1.9555283e-01
      79	 1.2735759e+00	 1.4294732e-01	 1.3253969e+00	 1.3931321e-01	[ 1.2558050e+00]	 1.9306564e-01
      80	 1.2795611e+00	 1.4238329e-01	 1.3313497e+00	 1.3907348e-01	[ 1.2563715e+00]	 1.9956803e-01
      81	 1.2837084e+00	 1.4188104e-01	 1.3355367e+00	 1.3839550e-01	[ 1.2599129e+00]	 2.0285368e-01
      82	 1.2894031e+00	 1.4137071e-01	 1.3413096e+00	 1.3771509e-01	[ 1.2661632e+00]	 1.8657994e-01
      83	 1.2943360e+00	 1.4081394e-01	 1.3463982e+00	 1.3674507e-01	[ 1.2706879e+00]	 2.1219230e-01
      84	 1.2990352e+00	 1.4049036e-01	 1.3511238e+00	 1.3630986e-01	[ 1.2755187e+00]	 1.9123197e-01
      85	 1.3055576e+00	 1.3971416e-01	 1.3577270e+00	 1.3575470e-01	[ 1.2834117e+00]	 1.8412304e-01
      86	 1.3094703e+00	 1.3979836e-01	 1.3616675e+00	 1.3586556e-01	[ 1.2864483e+00]	 1.8084526e-01
      87	 1.3133902e+00	 1.3934056e-01	 1.3655796e+00	 1.3568626e-01	[ 1.2901473e+00]	 1.8665791e-01
      88	 1.3192195e+00	 1.3814897e-01	 1.3715323e+00	 1.3526144e-01	[ 1.2962264e+00]	 1.8221736e-01
      89	 1.3223903e+00	 1.3738005e-01	 1.3748612e+00	 1.3503001e-01	[ 1.2985292e+00]	 2.0201230e-01
      90	 1.3258272e+00	 1.3691042e-01	 1.3783756e+00	 1.3479187e-01	[ 1.3008396e+00]	 1.9483924e-01
      91	 1.3303115e+00	 1.3613736e-01	 1.3830343e+00	 1.3450639e-01	[ 1.3052050e+00]	 1.8711400e-01
      92	 1.3329829e+00	 1.3593109e-01	 1.3857733e+00	 1.3446598e-01	  1.3046443e+00 	 1.9337320e-01
      93	 1.3383851e+00	 1.3509776e-01	 1.3913851e+00	 1.3431997e-01	[ 1.3067196e+00]	 1.8287659e-01
      94	 1.3412970e+00	 1.3462002e-01	 1.3944761e+00	 1.3440121e-01	  1.3051046e+00 	 2.0871258e-01
      95	 1.3446916e+00	 1.3455142e-01	 1.3977765e+00	 1.3417540e-01	[ 1.3092617e+00]	 2.1106267e-01
      96	 1.3471196e+00	 1.3438335e-01	 1.4001901e+00	 1.3397977e-01	[ 1.3139855e+00]	 1.9053006e-01
      97	 1.3497539e+00	 1.3423154e-01	 1.4029014e+00	 1.3364074e-01	[ 1.3176976e+00]	 1.8134880e-01
      98	 1.3524680e+00	 1.3412866e-01	 1.4059360e+00	 1.3327692e-01	[ 1.3190731e+00]	 1.9184780e-01
      99	 1.3563360e+00	 1.3371273e-01	 1.4098331e+00	 1.3288574e-01	[ 1.3209010e+00]	 1.8923068e-01
     100	 1.3590073e+00	 1.3339204e-01	 1.4125630e+00	 1.3268781e-01	  1.3198041e+00 	 1.9289136e-01
     101	 1.3623631e+00	 1.3290197e-01	 1.4160484e+00	 1.3246870e-01	  1.3172828e+00 	 1.8197203e-01
     102	 1.3665869e+00	 1.3255716e-01	 1.4204570e+00	 1.3219149e-01	  1.3156829e+00 	 1.7921758e-01
     103	 1.3708684e+00	 1.3192996e-01	 1.4248856e+00	 1.3196307e-01	  1.3134909e+00 	 1.8515038e-01
     104	 1.3734755e+00	 1.3193506e-01	 1.4273528e+00	 1.3205205e-01	  1.3174843e+00 	 1.8143511e-01
     105	 1.3766741e+00	 1.3179749e-01	 1.4304732e+00	 1.3229581e-01	[ 1.3217362e+00]	 2.0032716e-01
     106	 1.3793046e+00	 1.3140057e-01	 1.4331184e+00	 1.3266904e-01	  1.3184048e+00 	 1.7945910e-01
     107	 1.3817893e+00	 1.3111947e-01	 1.4355852e+00	 1.3270346e-01	  1.3182093e+00 	 1.8400168e-01
     108	 1.3856093e+00	 1.3017826e-01	 1.4395285e+00	 1.3271019e-01	  1.3134497e+00 	 1.8398213e-01
     109	 1.3879009e+00	 1.2994069e-01	 1.4418501e+00	 1.3254886e-01	  1.3136852e+00 	 2.1493816e-01
     110	 1.3909625e+00	 1.2951929e-01	 1.4449008e+00	 1.3246327e-01	  1.3153776e+00 	 1.8979740e-01
     111	 1.3943603e+00	 1.2898577e-01	 1.4482744e+00	 1.3190956e-01	[ 1.3221855e+00]	 1.9100332e-01
     112	 1.3970502e+00	 1.2888768e-01	 1.4509046e+00	 1.3175265e-01	[ 1.3225482e+00]	 1.7734790e-01
     113	 1.3989684e+00	 1.2883141e-01	 1.4527731e+00	 1.3152623e-01	[ 1.3251201e+00]	 1.8713379e-01
     114	 1.4021579e+00	 1.2870795e-01	 1.4559920e+00	 1.3126775e-01	[ 1.3261365e+00]	 1.8634343e-01
     115	 1.4039268e+00	 1.2835611e-01	 1.4578795e+00	 1.3110432e-01	  1.3231472e+00 	 1.8030238e-01
     116	 1.4061133e+00	 1.2840777e-01	 1.4600216e+00	 1.3110039e-01	[ 1.3261773e+00]	 1.8479872e-01
     117	 1.4080164e+00	 1.2838815e-01	 1.4619684e+00	 1.3114567e-01	[ 1.3266738e+00]	 1.8523097e-01
     118	 1.4098453e+00	 1.2840724e-01	 1.4638611e+00	 1.3113195e-01	[ 1.3270647e+00]	 1.8409014e-01
     119	 1.4136575e+00	 1.2847790e-01	 1.4678670e+00	 1.3100724e-01	[ 1.3303055e+00]	 1.8451309e-01
     120	 1.4158503e+00	 1.2850354e-01	 1.4702083e+00	 1.3095890e-01	  1.3300224e+00 	 3.7216783e-01
     121	 1.4180807e+00	 1.2848190e-01	 1.4724742e+00	 1.3082349e-01	[ 1.3322729e+00]	 1.9445157e-01
     122	 1.4201293e+00	 1.2838443e-01	 1.4745623e+00	 1.3076428e-01	[ 1.3335107e+00]	 1.8682265e-01
     123	 1.4218210e+00	 1.2828794e-01	 1.4763250e+00	 1.3068584e-01	  1.3332093e+00 	 1.8079424e-01
     124	 1.4241000e+00	 1.2811169e-01	 1.4786926e+00	 1.3083275e-01	  1.3311337e+00 	 1.9325829e-01
     125	 1.4266681e+00	 1.2793804e-01	 1.4813614e+00	 1.3093016e-01	  1.3293614e+00 	 1.8940091e-01
     126	 1.4293597e+00	 1.2767140e-01	 1.4842092e+00	 1.3116284e-01	  1.3246997e+00 	 2.1201372e-01
     127	 1.4320812e+00	 1.2752908e-01	 1.4870496e+00	 1.3118344e-01	  1.3246472e+00 	 1.8097401e-01
     128	 1.4344632e+00	 1.2741902e-01	 1.4894333e+00	 1.3125337e-01	  1.3243123e+00 	 1.8077517e-01
     129	 1.4366270e+00	 1.2735146e-01	 1.4916108e+00	 1.3129693e-01	  1.3242241e+00 	 1.8250585e-01
     130	 1.4388344e+00	 1.2725650e-01	 1.4937672e+00	 1.3144121e-01	  1.3244787e+00 	 2.0355844e-01
     131	 1.4413862e+00	 1.2718292e-01	 1.4962516e+00	 1.3155814e-01	  1.3251479e+00 	 1.9638610e-01
     132	 1.4440305e+00	 1.2702672e-01	 1.4989004e+00	 1.3186523e-01	  1.3221739e+00 	 1.9656181e-01
     133	 1.4460200e+00	 1.2707742e-01	 1.5008724e+00	 1.3172078e-01	  1.3244669e+00 	 1.8892169e-01
     134	 1.4475150e+00	 1.2706557e-01	 1.5023662e+00	 1.3165494e-01	  1.3243963e+00 	 1.9293046e-01
     135	 1.4494168e+00	 1.2700355e-01	 1.5044365e+00	 1.3159860e-01	  1.3188530e+00 	 1.9016027e-01
     136	 1.4515186e+00	 1.2686138e-01	 1.5065073e+00	 1.3148290e-01	  1.3197580e+00 	 2.0568824e-01
     137	 1.4525639e+00	 1.2675852e-01	 1.5075365e+00	 1.3147581e-01	  1.3197654e+00 	 1.8392205e-01
     138	 1.4558061e+00	 1.2630367e-01	 1.5108452e+00	 1.3151954e-01	  1.3194314e+00 	 1.8800735e-01
     139	 1.4571748e+00	 1.2611909e-01	 1.5122586e+00	 1.3151703e-01	  1.3122512e+00 	 4.0306568e-01
     140	 1.4588722e+00	 1.2590635e-01	 1.5139959e+00	 1.3151750e-01	  1.3124927e+00 	 1.9651628e-01
     141	 1.4609654e+00	 1.2563260e-01	 1.5161500e+00	 1.3137032e-01	  1.3131361e+00 	 1.9286895e-01
     142	 1.4624513e+00	 1.2541398e-01	 1.5176658e+00	 1.3134345e-01	  1.3118749e+00 	 1.8541646e-01
     143	 1.4640815e+00	 1.2531617e-01	 1.5192684e+00	 1.3126456e-01	  1.3111291e+00 	 1.8788767e-01
     144	 1.4666352e+00	 1.2516378e-01	 1.5218316e+00	 1.3116196e-01	  1.3086584e+00 	 1.8745351e-01
     145	 1.4686139e+00	 1.2498460e-01	 1.5238915e+00	 1.3121492e-01	  1.3049028e+00 	 1.9314790e-01
     146	 1.4704557e+00	 1.2476427e-01	 1.5258555e+00	 1.3123873e-01	  1.3020582e+00 	 2.0475650e-01
     147	 1.4717876e+00	 1.2468287e-01	 1.5272326e+00	 1.3128588e-01	  1.3013131e+00 	 1.7632127e-01
     148	 1.4737109e+00	 1.2456071e-01	 1.5292413e+00	 1.3135158e-01	  1.3014452e+00 	 1.7809439e-01
     149	 1.4750551e+00	 1.2433259e-01	 1.5307514e+00	 1.3163558e-01	  1.3003005e+00 	 1.8369746e-01
     150	 1.4767932e+00	 1.2430591e-01	 1.5324614e+00	 1.3151861e-01	  1.3012022e+00 	 2.0467710e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 48 s, sys: 11.7 s, total: 59.7 s
    Wall time: 30.2 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fb0112cc460>



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
    Process 0 running estimator on chunk 0 - 10000
    Process 0 estimating GPz PZ PDF for rows 0 - 10,000
    Inserting handle into data store.  output_gpz_run: inprogress_output_gpz_run.hdf5, gpz_run
    Process 0 running estimator on chunk 10000 - 20000
    Process 0 estimating GPz PZ PDF for rows 10,000 - 20,000
    Process 0 running estimator on chunk 20000 - 20449
    Process 0 estimating GPz PZ PDF for rows 20,000 - 20,449
    CPU times: user 830 ms, sys: 291 ms, total: 1.12 s
    Wall time: 676 ms


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




.. image:: ../../../docs/rendered/estimation_examples/GPz_estimation_example_files/../../../docs/rendered/estimation_examples/GPz_estimation_example_16_1.png


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




.. image:: ../../../docs/rendered/estimation_examples/GPz_estimation_example_files/../../../docs/rendered/estimation_examples/GPz_estimation_example_19_1.png

