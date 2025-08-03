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

    ngal: 10225
    training model...


.. parsed-literal::

    Iter	 logML/n 		 Train RMSE		 Train RMSE/n		 Valid RMSE		 Valid MLL		 Time    
       1	-3.3205280e-01	 3.1713723e-01	-3.2230693e-01	 3.3401765e-01	[-3.5555403e-01]	 4.6727586e-01


.. parsed-literal::

       2	-2.6081648e-01	 3.0611263e-01	-2.3643391e-01	 3.2159039e-01	[-2.8557176e-01]	 2.3162627e-01


.. parsed-literal::

       3	-2.1642713e-01	 2.8600130e-01	-1.7440255e-01	 3.0054816e-01	[-2.3497726e-01]	 2.9668736e-01


.. parsed-literal::

       4	-1.8877419e-01	 2.6227143e-01	-1.4876999e-01	 2.7305501e-01	[-2.1961141e-01]	 2.0871425e-01


.. parsed-literal::

       5	-9.4121934e-02	 2.5414869e-01	-5.7724766e-02	 2.6820929e-01	[-1.2603956e-01]	 2.1066165e-01
       6	-5.9999477e-02	 2.4872139e-01	-2.7970840e-02	 2.6232207e-01	[-8.1373245e-02]	 1.7967844e-01


.. parsed-literal::

       7	-4.1162410e-02	 2.4590385e-01	-1.6444577e-02	 2.6021152e-01	[-7.5461357e-02]	 2.1745086e-01


.. parsed-literal::

       8	-2.9979691e-02	 2.4415370e-01	-9.0525572e-03	 2.5870626e-01	[-7.0208486e-02]	 2.1770573e-01


.. parsed-literal::

       9	-1.6090989e-02	 2.4163943e-01	 1.7477525e-03	 2.5654166e-01	[-6.0955157e-02]	 2.1281981e-01


.. parsed-literal::

      10	-6.7849827e-03	 2.3983346e-01	 8.6424351e-03	 2.5465013e-01	[-6.0420297e-02]	 2.1864533e-01


.. parsed-literal::

      11	 3.6703496e-04	 2.3865212e-01	 1.5024997e-02	 2.5388505e-01	[-4.8577744e-02]	 2.0887542e-01


.. parsed-literal::

      12	 2.8350492e-03	 2.3821082e-01	 1.7284232e-02	 2.5349006e-01	[-4.8262626e-02]	 2.0191717e-01


.. parsed-literal::

      13	 7.6466099e-03	 2.3734264e-01	 2.1786369e-02	 2.5249676e-01	[-4.3788797e-02]	 2.1144533e-01
      14	 1.3848810e-02	 2.3601095e-01	 2.8908468e-02	 2.5152248e-01	[-4.0751874e-02]	 1.7901349e-01


.. parsed-literal::

      15	 1.6457290e-01	 2.2350808e-01	 1.8938677e-01	 2.4130102e-01	[ 1.2471926e-01]	 3.3414960e-01


.. parsed-literal::

      16	 2.0433324e-01	 2.1903489e-01	 2.3008548e-01	 2.3616845e-01	[ 1.8717400e-01]	 3.1694984e-01
      17	 2.4990861e-01	 2.1466421e-01	 2.7719269e-01	 2.3175396e-01	[ 2.2933906e-01]	 2.0008850e-01


.. parsed-literal::

      18	 3.1820752e-01	 2.0803187e-01	 3.5003888e-01	 2.2434053e-01	[ 2.8314547e-01]	 1.9684958e-01


.. parsed-literal::

      19	 3.6656462e-01	 2.0417793e-01	 4.0050679e-01	 2.2188472e-01	[ 3.2548052e-01]	 2.1679115e-01


.. parsed-literal::

      20	 4.2137597e-01	 2.0613575e-01	 4.5528297e-01	 2.2371065e-01	[ 4.0333162e-01]	 2.0447826e-01


.. parsed-literal::

      21	 4.7813993e-01	 2.0492609e-01	 5.1259977e-01	 2.2031701e-01	[ 4.6977059e-01]	 2.0829105e-01


.. parsed-literal::

      22	 5.2930677e-01	 2.0148419e-01	 5.6473816e-01	 2.1686614e-01	[ 5.2561536e-01]	 2.0629168e-01


.. parsed-literal::

      23	 5.9687927e-01	 1.9764149e-01	 6.3519888e-01	 2.1411712e-01	[ 6.0305491e-01]	 2.0564389e-01


.. parsed-literal::

      24	 6.1713692e-01	 2.0152924e-01	 6.5641473e-01	 2.1627691e-01	[ 6.3898325e-01]	 2.0639753e-01


.. parsed-literal::

      25	 6.5241390e-01	 1.9560804e-01	 6.8942932e-01	 2.1179700e-01	[ 6.6423150e-01]	 2.0381761e-01


.. parsed-literal::

      26	 6.7529845e-01	 1.9363666e-01	 7.1221841e-01	 2.1181319e-01	[ 6.8104454e-01]	 2.0461535e-01


.. parsed-literal::

      27	 6.9863401e-01	 1.9212852e-01	 7.3512755e-01	 2.1224732e-01	[ 6.9923773e-01]	 3.1130886e-01


.. parsed-literal::

      28	 7.1544678e-01	 1.9139274e-01	 7.5205914e-01	 2.1188446e-01	[ 7.1436547e-01]	 3.2328629e-01


.. parsed-literal::

      29	 7.4181024e-01	 1.9118684e-01	 7.7911744e-01	 2.1104674e-01	[ 7.3811448e-01]	 2.1514964e-01


.. parsed-literal::

      30	 7.7438152e-01	 1.8954362e-01	 8.1271145e-01	 2.0968620e-01	[ 7.8125721e-01]	 2.0858002e-01


.. parsed-literal::

      31	 8.0009213e-01	 1.9216270e-01	 8.3826152e-01	 2.1205214e-01	[ 8.0111438e-01]	 2.1274662e-01
      32	 8.1769440e-01	 1.9320635e-01	 8.5644590e-01	 2.1310309e-01	[ 8.2038546e-01]	 1.9936609e-01


.. parsed-literal::

      33	 8.3676949e-01	 1.9226765e-01	 8.7643856e-01	 2.1203878e-01	[ 8.4457391e-01]	 2.0325208e-01


.. parsed-literal::

      34	 8.3753986e-01	 1.9712421e-01	 8.7871573e-01	 2.1531550e-01	[ 8.5010913e-01]	 2.1886063e-01


.. parsed-literal::

      35	 8.7344650e-01	 1.9304493e-01	 9.1600740e-01	 2.1192129e-01	[ 8.9671721e-01]	 2.0409226e-01
      36	 8.9168956e-01	 1.8997730e-01	 9.3300591e-01	 2.0830779e-01	[ 9.0947656e-01]	 1.7298722e-01


.. parsed-literal::

      37	 9.0592322e-01	 1.8819633e-01	 9.4730673e-01	 2.0645343e-01	[ 9.2391092e-01]	 2.0599580e-01
      38	 9.2178056e-01	 1.8598413e-01	 9.6386326e-01	 2.0429294e-01	[ 9.3888074e-01]	 1.8304491e-01


.. parsed-literal::

      39	 9.3799077e-01	 1.8250222e-01	 9.8029407e-01	 2.0123861e-01	[ 9.5415761e-01]	 2.0582557e-01


.. parsed-literal::

      40	 9.5752870e-01	 1.7706028e-01	 1.0002453e+00	 1.9753494e-01	[ 9.7311511e-01]	 2.1181512e-01


.. parsed-literal::

      41	 9.7112899e-01	 1.7405219e-01	 1.0143960e+00	 1.9523247e-01	[ 9.8541458e-01]	 2.1016550e-01


.. parsed-literal::

      42	 9.8374624e-01	 1.7192536e-01	 1.0272095e+00	 1.9354100e-01	[ 9.9804205e-01]	 2.1030378e-01


.. parsed-literal::

      43	 9.9888119e-01	 1.7052999e-01	 1.0431069e+00	 1.9191527e-01	[ 1.0110065e+00]	 2.1031380e-01


.. parsed-literal::

      44	 1.0130875e+00	 1.6805549e-01	 1.0584142e+00	 1.8956066e-01	[ 1.0234924e+00]	 2.1708894e-01
      45	 1.0274464e+00	 1.6739446e-01	 1.0732924e+00	 1.8877148e-01	[ 1.0355182e+00]	 2.0257163e-01


.. parsed-literal::

      46	 1.0405306e+00	 1.6580223e-01	 1.0867668e+00	 1.8730203e-01	[ 1.0459310e+00]	 2.1545100e-01
      47	 1.0509505e+00	 1.6501486e-01	 1.0976323e+00	 1.8672151e-01	[ 1.0559544e+00]	 1.8793583e-01


.. parsed-literal::

      48	 1.0597787e+00	 1.6277990e-01	 1.1064941e+00	 1.8478707e-01	[ 1.0618286e+00]	 2.1427512e-01


.. parsed-literal::

      49	 1.0692089e+00	 1.6101072e-01	 1.1159342e+00	 1.8291592e-01	[ 1.0728522e+00]	 2.0696449e-01
      50	 1.0801467e+00	 1.5904212e-01	 1.1270527e+00	 1.8081424e-01	[ 1.0859394e+00]	 1.8661332e-01


.. parsed-literal::

      51	 1.0916175e+00	 1.5630936e-01	 1.1389061e+00	 1.7818460e-01	[ 1.0980244e+00]	 1.8284297e-01
      52	 1.1012120e+00	 1.5424543e-01	 1.1485042e+00	 1.7653807e-01	[ 1.1062876e+00]	 2.0227194e-01


.. parsed-literal::

      53	 1.1088325e+00	 1.5351560e-01	 1.1558436e+00	 1.7639325e-01	[ 1.1132417e+00]	 2.0725298e-01


.. parsed-literal::

      54	 1.1219805e+00	 1.5178412e-01	 1.1691888e+00	 1.7587423e-01	[ 1.1247154e+00]	 2.1162367e-01
      55	 1.1298748e+00	 1.5185344e-01	 1.1768613e+00	 1.7713331e-01	  1.1211710e+00 	 1.9279742e-01


.. parsed-literal::

      56	 1.1383381e+00	 1.5081685e-01	 1.1853846e+00	 1.7637976e-01	[ 1.1306855e+00]	 2.1671367e-01
      57	 1.1464433e+00	 1.4988655e-01	 1.1936855e+00	 1.7563857e-01	[ 1.1389766e+00]	 1.8701243e-01


.. parsed-literal::

      58	 1.1549651e+00	 1.4881159e-01	 1.2026994e+00	 1.7490464e-01	[ 1.1439641e+00]	 2.0211959e-01
      59	 1.1641319e+00	 1.4872613e-01	 1.2127691e+00	 1.7514943e-01	  1.1425079e+00 	 1.9969726e-01


.. parsed-literal::

      60	 1.1723920e+00	 1.4763241e-01	 1.2209978e+00	 1.7447326e-01	[ 1.1489627e+00]	 2.1441174e-01


.. parsed-literal::

      61	 1.1804698e+00	 1.4664363e-01	 1.2292402e+00	 1.7394057e-01	[ 1.1533060e+00]	 2.0837712e-01


.. parsed-literal::

      62	 1.1876705e+00	 1.4601252e-01	 1.2368005e+00	 1.7352010e-01	[ 1.1560902e+00]	 2.1138120e-01
      63	 1.1963078e+00	 1.4538288e-01	 1.2457107e+00	 1.7284816e-01	[ 1.1621178e+00]	 1.9786477e-01


.. parsed-literal::

      64	 1.2038881e+00	 1.4492639e-01	 1.2536595e+00	 1.7219349e-01	[ 1.1655974e+00]	 2.0352817e-01
      65	 1.2129900e+00	 1.4468024e-01	 1.2631060e+00	 1.7147379e-01	[ 1.1731139e+00]	 1.9852829e-01


.. parsed-literal::

      66	 1.2216179e+00	 1.4441569e-01	 1.2721630e+00	 1.7071658e-01	[ 1.1741846e+00]	 2.0423245e-01


.. parsed-literal::

      67	 1.2286583e+00	 1.4384821e-01	 1.2791530e+00	 1.6992091e-01	[ 1.1889404e+00]	 2.0816326e-01


.. parsed-literal::

      68	 1.2346077e+00	 1.4347768e-01	 1.2849027e+00	 1.6971508e-01	[ 1.1960716e+00]	 2.1522212e-01


.. parsed-literal::

      69	 1.2414221e+00	 1.4280953e-01	 1.2918645e+00	 1.6923637e-01	[ 1.2005080e+00]	 2.0662189e-01


.. parsed-literal::

      70	 1.2508669e+00	 1.4220531e-01	 1.3014721e+00	 1.6845142e-01	[ 1.2085704e+00]	 2.0939946e-01


.. parsed-literal::

      71	 1.2574363e+00	 1.4231777e-01	 1.3082881e+00	 1.6835092e-01	[ 1.2118061e+00]	 2.1403337e-01


.. parsed-literal::

      72	 1.2622599e+00	 1.4193147e-01	 1.3129308e+00	 1.6809241e-01	[ 1.2196068e+00]	 2.0547724e-01
      73	 1.2676882e+00	 1.4174820e-01	 1.3185488e+00	 1.6781685e-01	[ 1.2257559e+00]	 1.7770863e-01


.. parsed-literal::

      74	 1.2737542e+00	 1.4138384e-01	 1.3248151e+00	 1.6747109e-01	[ 1.2314288e+00]	 2.1884108e-01


.. parsed-literal::

      75	 1.2834853e+00	 1.3990797e-01	 1.3350746e+00	 1.6631493e-01	[ 1.2386320e+00]	 2.0992255e-01


.. parsed-literal::

      76	 1.2873584e+00	 1.3859492e-01	 1.3394537e+00	 1.6471169e-01	  1.2348181e+00 	 2.1154833e-01


.. parsed-literal::

      77	 1.2947664e+00	 1.3808740e-01	 1.3462652e+00	 1.6456396e-01	[ 1.2405820e+00]	 2.0257926e-01


.. parsed-literal::

      78	 1.2991513e+00	 1.3749373e-01	 1.3506084e+00	 1.6407393e-01	  1.2399885e+00 	 2.0784593e-01


.. parsed-literal::

      79	 1.3044826e+00	 1.3652160e-01	 1.3559852e+00	 1.6323857e-01	  1.2348098e+00 	 2.0908499e-01


.. parsed-literal::

      80	 1.3099752e+00	 1.3618964e-01	 1.3615724e+00	 1.6271998e-01	  1.2357970e+00 	 2.1322417e-01
      81	 1.3167707e+00	 1.3570509e-01	 1.3683041e+00	 1.6241638e-01	  1.2348530e+00 	 1.8348193e-01


.. parsed-literal::

      82	 1.3224628e+00	 1.3548405e-01	 1.3739510e+00	 1.6239884e-01	  1.2385754e+00 	 2.0638371e-01
      83	 1.3272922e+00	 1.3528540e-01	 1.3789202e+00	 1.6233672e-01	  1.2403794e+00 	 1.9839740e-01


.. parsed-literal::

      84	 1.3322866e+00	 1.3471042e-01	 1.3843131e+00	 1.6181344e-01	[ 1.2493670e+00]	 2.0003653e-01
      85	 1.3395537e+00	 1.3462888e-01	 1.3915043e+00	 1.6162532e-01	[ 1.2515774e+00]	 1.9931436e-01


.. parsed-literal::

      86	 1.3424573e+00	 1.3451463e-01	 1.3943687e+00	 1.6140857e-01	[ 1.2542107e+00]	 2.0688915e-01
      87	 1.3486158e+00	 1.3424733e-01	 1.4007776e+00	 1.6086661e-01	[ 1.2571983e+00]	 1.9799924e-01


.. parsed-literal::

      88	 1.3556722e+00	 1.3319154e-01	 1.4081767e+00	 1.5930908e-01	[ 1.2631918e+00]	 2.0201898e-01


.. parsed-literal::

      89	 1.3616453e+00	 1.3282715e-01	 1.4145123e+00	 1.5845590e-01	[ 1.2675002e+00]	 2.1246052e-01
      90	 1.3676314e+00	 1.3227969e-01	 1.4203896e+00	 1.5788812e-01	[ 1.2721128e+00]	 1.9665408e-01


.. parsed-literal::

      91	 1.3747332e+00	 1.3151158e-01	 1.4277213e+00	 1.5707745e-01	[ 1.2726463e+00]	 2.2234344e-01


.. parsed-literal::

      92	 1.3790163e+00	 1.3128803e-01	 1.4321116e+00	 1.5691494e-01	[ 1.2770134e+00]	 2.1211481e-01


.. parsed-literal::

      93	 1.3846298e+00	 1.3088629e-01	 1.4378856e+00	 1.5657479e-01	[ 1.2821005e+00]	 2.1484160e-01
      94	 1.3899807e+00	 1.3045128e-01	 1.4434854e+00	 1.5664451e-01	[ 1.2831673e+00]	 1.9345236e-01


.. parsed-literal::

      95	 1.3945503e+00	 1.2996218e-01	 1.4481335e+00	 1.5636748e-01	[ 1.2903688e+00]	 2.0797729e-01


.. parsed-literal::

      96	 1.3985890e+00	 1.2958758e-01	 1.4520743e+00	 1.5634884e-01	[ 1.2959589e+00]	 2.1497297e-01


.. parsed-literal::

      97	 1.4040718e+00	 1.2881768e-01	 1.4577964e+00	 1.5583224e-01	[ 1.2995816e+00]	 2.1245408e-01


.. parsed-literal::

      98	 1.4094258e+00	 1.2844007e-01	 1.4630156e+00	 1.5558545e-01	[ 1.3036501e+00]	 2.0489359e-01
      99	 1.4125236e+00	 1.2819669e-01	 1.4660585e+00	 1.5519563e-01	[ 1.3077410e+00]	 1.8861365e-01


.. parsed-literal::

     100	 1.4170247e+00	 1.2779735e-01	 1.4706238e+00	 1.5459370e-01	[ 1.3135064e+00]	 2.0938182e-01
     101	 1.4207152e+00	 1.2743070e-01	 1.4743248e+00	 1.5404192e-01	[ 1.3167491e+00]	 1.7709470e-01


.. parsed-literal::

     102	 1.4238243e+00	 1.2726126e-01	 1.4773955e+00	 1.5397824e-01	[ 1.3215074e+00]	 2.0718312e-01


.. parsed-literal::

     103	 1.4288783e+00	 1.2676373e-01	 1.4825921e+00	 1.5365577e-01	[ 1.3263283e+00]	 2.0689416e-01


.. parsed-literal::

     104	 1.4319402e+00	 1.2645496e-01	 1.4856266e+00	 1.5333944e-01	[ 1.3332040e+00]	 2.0384097e-01
     105	 1.4364789e+00	 1.2591426e-01	 1.4901449e+00	 1.5266482e-01	[ 1.3351150e+00]	 1.9952393e-01


.. parsed-literal::

     106	 1.4416459e+00	 1.2554062e-01	 1.4953653e+00	 1.5246962e-01	  1.3346289e+00 	 1.7486668e-01


.. parsed-literal::

     107	 1.4453703e+00	 1.2531088e-01	 1.4990549e+00	 1.5212723e-01	[ 1.3352324e+00]	 2.1807909e-01
     108	 1.4488868e+00	 1.2514372e-01	 1.5024866e+00	 1.5231571e-01	[ 1.3362019e+00]	 1.8589687e-01


.. parsed-literal::

     109	 1.4529452e+00	 1.2503160e-01	 1.5066501e+00	 1.5251662e-01	[ 1.3365455e+00]	 2.1122217e-01


.. parsed-literal::

     110	 1.4562209e+00	 1.2434554e-01	 1.5101511e+00	 1.5217794e-01	[ 1.3372873e+00]	 2.1719170e-01


.. parsed-literal::

     111	 1.4596735e+00	 1.2394283e-01	 1.5136184e+00	 1.5186339e-01	[ 1.3395154e+00]	 2.1007752e-01


.. parsed-literal::

     112	 1.4634916e+00	 1.2334695e-01	 1.5175632e+00	 1.5138901e-01	  1.3393111e+00 	 2.1646500e-01


.. parsed-literal::

     113	 1.4665153e+00	 1.2301464e-01	 1.5206914e+00	 1.5137958e-01	  1.3381299e+00 	 2.1135616e-01


.. parsed-literal::

     114	 1.4699060e+00	 1.2259033e-01	 1.5243176e+00	 1.5142212e-01	  1.3309019e+00 	 2.0489883e-01
     115	 1.4729739e+00	 1.2240842e-01	 1.5274646e+00	 1.5160569e-01	  1.3272848e+00 	 2.0158315e-01


.. parsed-literal::

     116	 1.4761012e+00	 1.2232290e-01	 1.5306178e+00	 1.5185722e-01	  1.3242724e+00 	 1.8027592e-01
     117	 1.4797641e+00	 1.2206563e-01	 1.5344095e+00	 1.5203084e-01	  1.3174195e+00 	 1.9227028e-01


.. parsed-literal::

     118	 1.4817728e+00	 1.2198559e-01	 1.5365297e+00	 1.5261137e-01	  1.3067985e+00 	 2.1177053e-01


.. parsed-literal::

     119	 1.4848535e+00	 1.2188826e-01	 1.5394357e+00	 1.5240805e-01	  1.3141520e+00 	 2.1385217e-01
     120	 1.4864237e+00	 1.2173141e-01	 1.5410043e+00	 1.5231166e-01	  1.3150778e+00 	 1.8842983e-01


.. parsed-literal::

     121	 1.4892527e+00	 1.2158243e-01	 1.5438524e+00	 1.5230748e-01	  1.3158477e+00 	 2.1021938e-01


.. parsed-literal::

     122	 1.4930945e+00	 1.2128153e-01	 1.5478572e+00	 1.5231265e-01	  1.3097968e+00 	 2.0268703e-01
     123	 1.4954870e+00	 1.2106941e-01	 1.5503997e+00	 1.5192337e-01	  1.3135679e+00 	 1.7976570e-01


.. parsed-literal::

     124	 1.4982258e+00	 1.2107147e-01	 1.5530214e+00	 1.5203255e-01	  1.3139421e+00 	 2.0412707e-01


.. parsed-literal::

     125	 1.5000794e+00	 1.2102511e-01	 1.5549007e+00	 1.5203556e-01	  1.3099079e+00 	 2.0684290e-01


.. parsed-literal::

     126	 1.5023469e+00	 1.2086812e-01	 1.5572071e+00	 1.5183866e-01	  1.3080323e+00 	 2.1350265e-01


.. parsed-literal::

     127	 1.5038759e+00	 1.2078299e-01	 1.5587878e+00	 1.5163621e-01	  1.3071663e+00 	 3.2705975e-01


.. parsed-literal::

     128	 1.5056036e+00	 1.2059605e-01	 1.5605580e+00	 1.5141416e-01	  1.3052028e+00 	 2.0860481e-01


.. parsed-literal::

     129	 1.5070691e+00	 1.2046230e-01	 1.5620550e+00	 1.5122960e-01	  1.3034416e+00 	 2.1149731e-01


.. parsed-literal::

     130	 1.5099772e+00	 1.2024371e-01	 1.5650488e+00	 1.5084053e-01	  1.2969806e+00 	 2.0788240e-01


.. parsed-literal::

     131	 1.5114251e+00	 1.2021651e-01	 1.5665886e+00	 1.5071184e-01	  1.2944976e+00 	 3.1987786e-01


.. parsed-literal::

     132	 1.5134978e+00	 1.2016723e-01	 1.5687132e+00	 1.5052733e-01	  1.2888774e+00 	 2.1158934e-01


.. parsed-literal::

     133	 1.5154198e+00	 1.2013466e-01	 1.5707194e+00	 1.5041665e-01	  1.2854969e+00 	 2.1016788e-01


.. parsed-literal::

     134	 1.5171033e+00	 1.2016659e-01	 1.5725011e+00	 1.5035987e-01	  1.2830338e+00 	 2.0959640e-01


.. parsed-literal::

     135	 1.5187707e+00	 1.2011864e-01	 1.5742056e+00	 1.5027229e-01	  1.2806508e+00 	 2.1378756e-01


.. parsed-literal::

     136	 1.5205032e+00	 1.2003880e-01	 1.5759667e+00	 1.5026640e-01	  1.2787525e+00 	 2.0732284e-01


.. parsed-literal::

     137	 1.5220820e+00	 1.1981907e-01	 1.5776034e+00	 1.5014598e-01	  1.2785733e+00 	 2.1138167e-01


.. parsed-literal::

     138	 1.5236651e+00	 1.1974312e-01	 1.5791221e+00	 1.5028080e-01	  1.2743778e+00 	 2.1516109e-01


.. parsed-literal::

     139	 1.5247411e+00	 1.1965876e-01	 1.5801986e+00	 1.5030625e-01	  1.2701480e+00 	 2.1345901e-01


.. parsed-literal::

     140	 1.5263482e+00	 1.1951128e-01	 1.5818454e+00	 1.5034983e-01	  1.2637740e+00 	 2.1493340e-01


.. parsed-literal::

     141	 1.5274254e+00	 1.1937236e-01	 1.5829768e+00	 1.5032857e-01	  1.2590067e+00 	 3.2713199e-01
     142	 1.5289804e+00	 1.1924961e-01	 1.5845816e+00	 1.5031204e-01	  1.2564936e+00 	 1.7789650e-01


.. parsed-literal::

     143	 1.5307241e+00	 1.1912451e-01	 1.5863967e+00	 1.5027679e-01	  1.2566421e+00 	 2.1005607e-01


.. parsed-literal::

     144	 1.5321667e+00	 1.1901064e-01	 1.5878977e+00	 1.5014911e-01	  1.2562002e+00 	 2.0403385e-01


.. parsed-literal::

     145	 1.5334900e+00	 1.1893239e-01	 1.5892734e+00	 1.5010463e-01	  1.2569949e+00 	 2.0899796e-01
     146	 1.5348258e+00	 1.1887968e-01	 1.5905904e+00	 1.4994926e-01	  1.2577318e+00 	 1.9800305e-01


.. parsed-literal::

     147	 1.5363696e+00	 1.1878052e-01	 1.5921340e+00	 1.4970452e-01	  1.2547507e+00 	 2.0972109e-01


.. parsed-literal::

     148	 1.5376968e+00	 1.1874594e-01	 1.5935103e+00	 1.4950199e-01	  1.2530173e+00 	 2.1278405e-01


.. parsed-literal::

     149	 1.5392648e+00	 1.1867307e-01	 1.5951105e+00	 1.4920349e-01	  1.2499318e+00 	 2.0410585e-01
     150	 1.5404458e+00	 1.1861736e-01	 1.5963178e+00	 1.4911699e-01	  1.2468994e+00 	 1.8988752e-01


.. parsed-literal::

    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 7s, sys: 1.1 s, total: 2min 8s
    Wall time: 32.2 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f5640a70b50>



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
    Process 0 running estimator on chunk 10,000 - 20,000
    Process 0 estimating GPz PZ PDF for rows 10,000 - 20,000


.. parsed-literal::

    Process 0 running estimator on chunk 20,000 - 20,449
    Process 0 estimating GPz PZ PDF for rows 20,000 - 20,449
    CPU times: user 2.16 s, sys: 46 ms, total: 2.2 s
    Wall time: 669 ms


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

