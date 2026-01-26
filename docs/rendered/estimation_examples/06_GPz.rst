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
       1	-3.2739583e-01	 3.1584970e-01	-3.1757926e-01	 3.4134980e-01	[-3.6668831e-01]	 6.1530185e-01


.. parsed-literal::

       2	-2.5903169e-01	 3.0584561e-01	-2.3529617e-01	 3.2795618e-01	[-3.0602565e-01]	 3.0251765e-01


.. parsed-literal::

       3	-2.1660780e-01	 2.8630443e-01	-1.7567886e-01	 3.0581033e-01	[-2.5300682e-01]	 3.4769464e-01


.. parsed-literal::

       4	-1.8307728e-01	 2.6434712e-01	-1.4175312e-01	 2.7561583e-01	[-2.0161166e-01]	 2.1536660e-01


.. parsed-literal::

       5	-1.0055425e-01	 2.5601771e-01	-6.6625378e-02	 2.6445914e-01	[-1.0476917e-01]	 2.1562052e-01


.. parsed-literal::

       6	-6.4134831e-02	 2.5016410e-01	-3.4429330e-02	 2.5670949e-01	[-5.9161283e-02]	 2.1259189e-01


.. parsed-literal::

       7	-4.8064562e-02	 2.4771373e-01	-2.3806104e-02	 2.5390185e-01	[-4.9036858e-02]	 2.1990585e-01


.. parsed-literal::

       8	-3.3860991e-02	 2.4528542e-01	-1.3660494e-02	 2.5214312e-01	[-4.2475260e-02]	 2.2226667e-01


.. parsed-literal::

       9	-1.9202452e-02	 2.4250905e-01	-1.8018103e-03	 2.5038248e-01	[-3.5080580e-02]	 2.2782612e-01


.. parsed-literal::

      10	-6.3067376e-03	 2.4001131e-01	 8.5880057e-03	 2.4977382e-01	[-3.4015148e-02]	 2.1495390e-01


.. parsed-literal::

      11	-2.3523955e-03	 2.3940090e-01	 1.1408801e-02	 2.4936022e-01	[-2.7105834e-02]	 2.1439648e-01


.. parsed-literal::

      12	 1.2917372e-03	 2.3883400e-01	 1.4873007e-02	 2.4872494e-01	[-2.6338099e-02]	 2.2738028e-01


.. parsed-literal::

      13	 4.3323430e-03	 2.3825315e-01	 1.7734439e-02	 2.4859114e-01	[-2.5718879e-02]	 2.2631693e-01


.. parsed-literal::

      14	 9.0224074e-03	 2.3729262e-01	 2.2941442e-02	 2.4819672e-01	[-2.3542722e-02]	 2.1436548e-01


.. parsed-literal::

      15	 1.6267269e-01	 2.2260135e-01	 1.8680315e-01	 2.3176816e-01	[ 1.8054403e-01]	 3.8318253e-01


.. parsed-literal::

      16	 2.5603033e-01	 2.1841628e-01	 2.8197228e-01	 2.2822107e-01	[ 2.6845062e-01]	 2.4167180e-01


.. parsed-literal::

      17	 2.7688974e-01	 2.1485863e-01	 3.0836177e-01	 2.2783581e-01	[ 2.8171038e-01]	 2.3752713e-01


.. parsed-literal::

      18	 3.4080932e-01	 2.1424573e-01	 3.7142480e-01	 2.2312486e-01	[ 3.5568512e-01]	 2.1766567e-01


.. parsed-literal::

      19	 3.7147370e-01	 2.1052620e-01	 4.0359626e-01	 2.2047177e-01	[ 3.8615006e-01]	 2.3737574e-01


.. parsed-literal::

      20	 4.2321507e-01	 2.0575580e-01	 4.5616544e-01	 2.1666975e-01	[ 4.3504537e-01]	 2.2623849e-01


.. parsed-literal::

      21	 4.9488329e-01	 2.0356247e-01	 5.2836816e-01	 2.1463339e-01	[ 5.0955847e-01]	 2.2507787e-01


.. parsed-literal::

      22	 5.6030158e-01	 2.0379556e-01	 5.9774194e-01	 2.1628346e-01	[ 5.6134131e-01]	 2.2661996e-01


.. parsed-literal::

      23	 5.9645592e-01	 2.0088855e-01	 6.3591703e-01	 2.1290926e-01	[ 5.9091764e-01]	 2.3494911e-01


.. parsed-literal::

      24	 6.2166698e-01	 1.9859705e-01	 6.6031322e-01	 2.1066189e-01	[ 6.1886858e-01]	 2.1903563e-01


.. parsed-literal::

      25	 6.5486975e-01	 1.9790300e-01	 6.9315771e-01	 2.1127005e-01	[ 6.5239832e-01]	 2.2791934e-01


.. parsed-literal::

      26	 6.7850091e-01	 2.0073106e-01	 7.1620729e-01	 2.1528361e-01	[ 6.7474923e-01]	 2.3872519e-01


.. parsed-literal::

      27	 7.0068768e-01	 1.9983970e-01	 7.3855656e-01	 2.1140788e-01	[ 6.9759156e-01]	 2.3100352e-01


.. parsed-literal::

      28	 7.2331675e-01	 2.0117619e-01	 7.6222895e-01	 2.1474985e-01	[ 7.1404501e-01]	 2.0767498e-01


.. parsed-literal::

      29	 7.3798399e-01	 2.0254113e-01	 7.7625522e-01	 2.1575757e-01	[ 7.4196375e-01]	 2.1605897e-01


.. parsed-literal::

      30	 7.5758981e-01	 2.0011056e-01	 7.9626628e-01	 2.1436263e-01	[ 7.5948561e-01]	 2.2539449e-01


.. parsed-literal::

      31	 7.6999747e-01	 1.9775812e-01	 8.0858104e-01	 2.1178390e-01	[ 7.6942085e-01]	 2.3002672e-01


.. parsed-literal::

      32	 7.8932402e-01	 1.9536531e-01	 8.2853391e-01	 2.1059287e-01	[ 7.8353900e-01]	 2.3707342e-01


.. parsed-literal::

      33	 8.1452716e-01	 1.9636002e-01	 8.5331306e-01	 2.1663409e-01	[ 8.0797463e-01]	 2.2893763e-01


.. parsed-literal::

      34	 8.4169202e-01	 1.9282650e-01	 8.8189968e-01	 2.1018814e-01	[ 8.4546366e-01]	 2.2211480e-01


.. parsed-literal::

      35	 8.6349793e-01	 1.9280784e-01	 9.0433593e-01	 2.0975671e-01	[ 8.5997252e-01]	 2.4033165e-01


.. parsed-literal::

      36	 8.7751358e-01	 1.9206373e-01	 9.1846398e-01	 2.0911470e-01	[ 8.7078145e-01]	 2.3702264e-01


.. parsed-literal::

      37	 8.9830827e-01	 1.9039812e-01	 9.4005744e-01	 2.0801934e-01	[ 8.9335828e-01]	 2.3999119e-01


.. parsed-literal::

      38	 9.0678832e-01	 1.8860891e-01	 9.4882461e-01	 2.0304361e-01	[ 8.9874147e-01]	 2.3103905e-01


.. parsed-literal::

      39	 9.2243726e-01	 1.8532259e-01	 9.6398460e-01	 2.0019910e-01	[ 9.1731965e-01]	 2.0581102e-01


.. parsed-literal::

      40	 9.3579341e-01	 1.8261222e-01	 9.7737832e-01	 1.9849592e-01	[ 9.2809473e-01]	 2.3174357e-01


.. parsed-literal::

      41	 9.4714565e-01	 1.8028546e-01	 9.8898086e-01	 1.9639258e-01	[ 9.3479070e-01]	 2.3577356e-01


.. parsed-literal::

      42	 9.6885310e-01	 1.7752854e-01	 1.0121208e+00	 1.9402168e-01	[ 9.4068364e-01]	 2.3645020e-01


.. parsed-literal::

      43	 9.8446242e-01	 1.7577747e-01	 1.0276370e+00	 1.9213852e-01	[ 9.5368251e-01]	 2.2468090e-01


.. parsed-literal::

      44	 9.9509684e-01	 1.7517088e-01	 1.0383354e+00	 1.8956021e-01	[ 9.6715600e-01]	 2.2326636e-01


.. parsed-literal::

      45	 1.0088461e+00	 1.7394029e-01	 1.0525846e+00	 1.8791124e-01	[ 9.8187847e-01]	 2.1209812e-01


.. parsed-literal::

      46	 1.0194829e+00	 1.7389687e-01	 1.0643541e+00	 1.8727582e-01	[ 9.8421210e-01]	 2.2994757e-01


.. parsed-literal::

      47	 1.0300905e+00	 1.7239832e-01	 1.0754427e+00	 1.8557753e-01	[ 9.9351457e-01]	 2.2995067e-01


.. parsed-literal::

      48	 1.0438086e+00	 1.7150496e-01	 1.0897173e+00	 1.8523368e-01	[ 1.0052350e+00]	 2.3165917e-01


.. parsed-literal::

      49	 1.0547455e+00	 1.7137883e-01	 1.1012731e+00	 1.8625058e-01	[ 1.0111889e+00]	 2.3866248e-01


.. parsed-literal::

      50	 1.0640974e+00	 1.6995597e-01	 1.1109396e+00	 1.8470636e-01	[ 1.0215764e+00]	 2.2569585e-01


.. parsed-literal::

      51	 1.0724993e+00	 1.6906480e-01	 1.1189981e+00	 1.8329200e-01	[ 1.0307816e+00]	 2.2160077e-01


.. parsed-literal::

      52	 1.0844920e+00	 1.6716652e-01	 1.1312270e+00	 1.8069601e-01	[ 1.0394364e+00]	 2.2981644e-01


.. parsed-literal::

      53	 1.0899158e+00	 1.6662398e-01	 1.1373635e+00	 1.7950338e-01	[ 1.0447832e+00]	 2.1852493e-01


.. parsed-literal::

      54	 1.0983337e+00	 1.6579209e-01	 1.1456461e+00	 1.7877552e-01	[ 1.0499999e+00]	 2.1657801e-01


.. parsed-literal::

      55	 1.1041822e+00	 1.6494177e-01	 1.1516527e+00	 1.7787922e-01	[ 1.0555976e+00]	 2.2069263e-01


.. parsed-literal::

      56	 1.1110737e+00	 1.6380054e-01	 1.1587351e+00	 1.7615997e-01	[ 1.0653866e+00]	 2.3153949e-01


.. parsed-literal::

      57	 1.1240579e+00	 1.6188419e-01	 1.1718118e+00	 1.7235231e-01	[ 1.0846275e+00]	 2.3686361e-01


.. parsed-literal::

      58	 1.1364311e+00	 1.5952473e-01	 1.1846338e+00	 1.6674969e-01	[ 1.1048791e+00]	 2.2410536e-01


.. parsed-literal::

      59	 1.1472569e+00	 1.5931224e-01	 1.1951897e+00	 1.6660820e-01	[ 1.1135468e+00]	 2.3270822e-01


.. parsed-literal::

      60	 1.1538021e+00	 1.5897125e-01	 1.2016159e+00	 1.6675532e-01	[ 1.1185864e+00]	 2.2420597e-01


.. parsed-literal::

      61	 1.1652170e+00	 1.5877768e-01	 1.2133770e+00	 1.6837548e-01	[ 1.1234416e+00]	 2.3429775e-01


.. parsed-literal::

      62	 1.1706332e+00	 1.5773185e-01	 1.2190539e+00	 1.6716690e-01	[ 1.1317247e+00]	 3.9378738e-01


.. parsed-literal::

      63	 1.1773333e+00	 1.5681532e-01	 1.2259462e+00	 1.6625725e-01	[ 1.1383537e+00]	 2.3668313e-01


.. parsed-literal::

      64	 1.1873834e+00	 1.5496311e-01	 1.2364959e+00	 1.6371129e-01	[ 1.1513868e+00]	 2.3114300e-01


.. parsed-literal::

      65	 1.1970725e+00	 1.5389110e-01	 1.2465945e+00	 1.6242913e-01	[ 1.1616708e+00]	 2.2610664e-01


.. parsed-literal::

      66	 1.2066286e+00	 1.5293838e-01	 1.2563647e+00	 1.6143971e-01	[ 1.1753853e+00]	 2.2547269e-01


.. parsed-literal::

      67	 1.2166122e+00	 1.5237654e-01	 1.2664399e+00	 1.6211510e-01	[ 1.1820759e+00]	 2.3678613e-01


.. parsed-literal::

      68	 1.2224242e+00	 1.5238058e-01	 1.2725349e+00	 1.6172970e-01	[ 1.1855184e+00]	 2.2544241e-01


.. parsed-literal::

      69	 1.2294864e+00	 1.5206279e-01	 1.2795553e+00	 1.6120439e-01	[ 1.1916505e+00]	 2.1404624e-01


.. parsed-literal::

      70	 1.2383807e+00	 1.5171196e-01	 1.2886750e+00	 1.5973498e-01	[ 1.1997402e+00]	 2.2235155e-01


.. parsed-literal::

      71	 1.2447962e+00	 1.5126382e-01	 1.2951372e+00	 1.5884286e-01	[ 1.2069619e+00]	 2.2576737e-01


.. parsed-literal::

      72	 1.2571296e+00	 1.5059714e-01	 1.3079154e+00	 1.5776863e-01	[ 1.2117228e+00]	 2.1394014e-01


.. parsed-literal::

      73	 1.2597135e+00	 1.4991183e-01	 1.3107050e+00	 1.5580551e-01	[ 1.2300879e+00]	 2.1975565e-01


.. parsed-literal::

      74	 1.2647472e+00	 1.4917608e-01	 1.3154464e+00	 1.5626235e-01	  1.2232645e+00 	 2.1384406e-01


.. parsed-literal::

      75	 1.2690948e+00	 1.4911197e-01	 1.3197225e+00	 1.5616709e-01	  1.2267246e+00 	 2.3235226e-01


.. parsed-literal::

      76	 1.2722171e+00	 1.4885928e-01	 1.3229481e+00	 1.5566508e-01	[ 1.2304332e+00]	 2.1502733e-01


.. parsed-literal::

      77	 1.2790549e+00	 1.4798626e-01	 1.3301652e+00	 1.5411561e-01	[ 1.2367528e+00]	 2.3496914e-01


.. parsed-literal::

      78	 1.2842467e+00	 1.4734305e-01	 1.3354934e+00	 1.5304568e-01	[ 1.2395992e+00]	 2.2845769e-01


.. parsed-literal::

      79	 1.2891070e+00	 1.4677857e-01	 1.3405930e+00	 1.5188553e-01	[ 1.2442381e+00]	 3.7660432e-01


.. parsed-literal::

      80	 1.2945169e+00	 1.4638278e-01	 1.3460263e+00	 1.5165002e-01	[ 1.2478497e+00]	 2.2771406e-01


.. parsed-literal::

      81	 1.2984132e+00	 1.4632768e-01	 1.3499459e+00	 1.5194840e-01	[ 1.2521559e+00]	 2.1197414e-01


.. parsed-literal::

      82	 1.3056213e+00	 1.4668024e-01	 1.3573934e+00	 1.5380862e-01	[ 1.2630030e+00]	 2.2772670e-01


.. parsed-literal::

      83	 1.3116037e+00	 1.4697150e-01	 1.3635582e+00	 1.5634226e-01	  1.2629880e+00 	 2.3459315e-01


.. parsed-literal::

      84	 1.3165566e+00	 1.4685064e-01	 1.3684972e+00	 1.5631407e-01	[ 1.2693312e+00]	 2.1550488e-01


.. parsed-literal::

      85	 1.3233458e+00	 1.4644379e-01	 1.3753846e+00	 1.5615762e-01	[ 1.2781458e+00]	 2.2741342e-01


.. parsed-literal::

      86	 1.3263649e+00	 1.4625189e-01	 1.3785230e+00	 1.5678800e-01	  1.2746458e+00 	 2.2876287e-01


.. parsed-literal::

      87	 1.3307626e+00	 1.4588957e-01	 1.3828820e+00	 1.5610840e-01	[ 1.2792791e+00]	 2.2843623e-01


.. parsed-literal::

      88	 1.3348313e+00	 1.4554888e-01	 1.3869780e+00	 1.5554633e-01	[ 1.2813741e+00]	 2.3938251e-01


.. parsed-literal::

      89	 1.3372779e+00	 1.4541858e-01	 1.3894108e+00	 1.5526641e-01	[ 1.2818710e+00]	 2.2758675e-01


.. parsed-literal::

      90	 1.3449057e+00	 1.4526647e-01	 1.3972273e+00	 1.5390996e-01	[ 1.2864016e+00]	 2.2963548e-01


.. parsed-literal::

      91	 1.3466159e+00	 1.4583009e-01	 1.3994835e+00	 1.5352534e-01	  1.2738482e+00 	 2.3027778e-01


.. parsed-literal::

      92	 1.3523115e+00	 1.4556886e-01	 1.4049440e+00	 1.5309162e-01	  1.2848966e+00 	 2.1365404e-01


.. parsed-literal::

      93	 1.3541933e+00	 1.4544743e-01	 1.4068544e+00	 1.5281655e-01	[ 1.2882075e+00]	 2.3300958e-01


.. parsed-literal::

      94	 1.3567958e+00	 1.4535060e-01	 1.4095174e+00	 1.5228479e-01	[ 1.2901126e+00]	 2.3056722e-01


.. parsed-literal::

      95	 1.3603568e+00	 1.4482953e-01	 1.4131878e+00	 1.5082407e-01	[ 1.2930215e+00]	 2.0363808e-01


.. parsed-literal::

      96	 1.3644598e+00	 1.4468419e-01	 1.4172936e+00	 1.4936056e-01	[ 1.2950379e+00]	 2.1591973e-01


.. parsed-literal::

      97	 1.3677011e+00	 1.4454302e-01	 1.4204807e+00	 1.4850106e-01	[ 1.2968068e+00]	 2.2399306e-01


.. parsed-literal::

      98	 1.3717121e+00	 1.4457726e-01	 1.4245831e+00	 1.4813202e-01	  1.2951284e+00 	 2.1323037e-01


.. parsed-literal::

      99	 1.3733462e+00	 1.4460139e-01	 1.4265350e+00	 1.4763961e-01	  1.2912478e+00 	 2.2505116e-01


.. parsed-literal::

     100	 1.3769658e+00	 1.4472449e-01	 1.4300945e+00	 1.4843139e-01	  1.2928815e+00 	 2.2749853e-01


.. parsed-literal::

     101	 1.3789840e+00	 1.4483187e-01	 1.4322030e+00	 1.4902776e-01	  1.2925547e+00 	 2.0587516e-01


.. parsed-literal::

     102	 1.3817146e+00	 1.4495023e-01	 1.4350683e+00	 1.4948805e-01	  1.2933112e+00 	 2.1643186e-01


.. parsed-literal::

     103	 1.3853662e+00	 1.4509392e-01	 1.4389203e+00	 1.4985511e-01	  1.2959525e+00 	 2.2745442e-01


.. parsed-literal::

     104	 1.3864638e+00	 1.4503478e-01	 1.4400567e+00	 1.4925558e-01	[ 1.3078954e+00]	 2.2751689e-01


.. parsed-literal::

     105	 1.3909840e+00	 1.4485207e-01	 1.4444346e+00	 1.4910214e-01	[ 1.3101553e+00]	 2.2696638e-01


.. parsed-literal::

     106	 1.3925368e+00	 1.4474617e-01	 1.4459484e+00	 1.4888633e-01	[ 1.3119445e+00]	 2.1811461e-01


.. parsed-literal::

     107	 1.3955835e+00	 1.4454525e-01	 1.4490281e+00	 1.4884231e-01	[ 1.3160159e+00]	 2.2011733e-01


.. parsed-literal::

     108	 1.3990468e+00	 1.4439781e-01	 1.4526450e+00	 1.4900033e-01	[ 1.3193691e+00]	 2.2673559e-01


.. parsed-literal::

     109	 1.4024363e+00	 1.4420247e-01	 1.4561308e+00	 1.4966451e-01	[ 1.3248041e+00]	 2.3973632e-01


.. parsed-literal::

     110	 1.4051191e+00	 1.4419671e-01	 1.4588164e+00	 1.5000389e-01	[ 1.3264310e+00]	 2.2571397e-01


.. parsed-literal::

     111	 1.4086716e+00	 1.4412818e-01	 1.4625113e+00	 1.5077214e-01	[ 1.3266738e+00]	 2.2326279e-01


.. parsed-literal::

     112	 1.4105617e+00	 1.4467448e-01	 1.4644726e+00	 1.5212797e-01	  1.3243394e+00 	 2.1586251e-01


.. parsed-literal::

     113	 1.4126094e+00	 1.4448282e-01	 1.4664814e+00	 1.5194399e-01	  1.3262248e+00 	 2.1018362e-01


.. parsed-literal::

     114	 1.4155097e+00	 1.4431102e-01	 1.4693456e+00	 1.5198729e-01	[ 1.3274966e+00]	 2.3157787e-01
     115	 1.4174071e+00	 1.4432420e-01	 1.4712191e+00	 1.5215411e-01	  1.3273642e+00 	 2.0535541e-01


.. parsed-literal::

     116	 1.4216600e+00	 1.4457906e-01	 1.4755674e+00	 1.5277571e-01	[ 1.3275760e+00]	 2.1559453e-01


.. parsed-literal::

     117	 1.4240741e+00	 1.4460556e-01	 1.4781517e+00	 1.5358234e-01	  1.3171817e+00 	 2.1809125e-01


.. parsed-literal::

     118	 1.4269707e+00	 1.4456952e-01	 1.4809349e+00	 1.5309701e-01	  1.3226581e+00 	 2.2708273e-01


.. parsed-literal::

     119	 1.4290456e+00	 1.4459860e-01	 1.4830934e+00	 1.5334008e-01	  1.3231690e+00 	 2.2912192e-01


.. parsed-literal::

     120	 1.4315200e+00	 1.4452722e-01	 1.4857227e+00	 1.5331831e-01	  1.3198957e+00 	 2.1582222e-01


.. parsed-literal::

     121	 1.4331781e+00	 1.4475059e-01	 1.4876647e+00	 1.5459651e-01	  1.3128888e+00 	 2.1774507e-01


.. parsed-literal::

     122	 1.4359293e+00	 1.4455461e-01	 1.4903281e+00	 1.5409290e-01	  1.3115797e+00 	 2.1266937e-01


.. parsed-literal::

     123	 1.4373987e+00	 1.4452935e-01	 1.4917216e+00	 1.5390905e-01	  1.3103967e+00 	 2.3865795e-01


.. parsed-literal::

     124	 1.4390243e+00	 1.4443856e-01	 1.4933308e+00	 1.5372231e-01	  1.3078816e+00 	 2.1733022e-01


.. parsed-literal::

     125	 1.4409614e+00	 1.4492503e-01	 1.4952908e+00	 1.5413321e-01	  1.3046092e+00 	 2.3493433e-01


.. parsed-literal::

     126	 1.4427040e+00	 1.4490177e-01	 1.4970447e+00	 1.5413905e-01	  1.3059684e+00 	 2.2837853e-01


.. parsed-literal::

     127	 1.4450579e+00	 1.4497306e-01	 1.4995131e+00	 1.5428764e-01	  1.3071194e+00 	 2.1825576e-01


.. parsed-literal::

     128	 1.4467336e+00	 1.4498854e-01	 1.5012330e+00	 1.5425626e-01	  1.3083990e+00 	 2.2352910e-01


.. parsed-literal::

     129	 1.4512395e+00	 1.4496261e-01	 1.5058578e+00	 1.5378263e-01	  1.3100047e+00 	 2.3543692e-01


.. parsed-literal::

     130	 1.4534921e+00	 1.4492575e-01	 1.5081732e+00	 1.5388476e-01	  1.3090378e+00 	 3.7713814e-01


.. parsed-literal::

     131	 1.4553963e+00	 1.4488127e-01	 1.5100202e+00	 1.5375028e-01	  1.3100162e+00 	 2.2628045e-01


.. parsed-literal::

     132	 1.4582354e+00	 1.4467081e-01	 1.5128396e+00	 1.5326166e-01	  1.3085411e+00 	 2.1533132e-01


.. parsed-literal::

     133	 1.4597465e+00	 1.4466246e-01	 1.5144580e+00	 1.5327340e-01	  1.3056892e+00 	 2.2066116e-01


.. parsed-literal::

     134	 1.4619264e+00	 1.4456213e-01	 1.5165812e+00	 1.5322084e-01	  1.3070624e+00 	 2.1237302e-01


.. parsed-literal::

     135	 1.4638322e+00	 1.4445445e-01	 1.5185208e+00	 1.5301646e-01	  1.3063462e+00 	 2.2096086e-01


.. parsed-literal::

     136	 1.4654435e+00	 1.4448968e-01	 1.5201706e+00	 1.5303679e-01	  1.3049310e+00 	 2.2318554e-01


.. parsed-literal::

     137	 1.4667106e+00	 1.4430231e-01	 1.5214812e+00	 1.5288407e-01	  1.3038473e+00 	 3.8622379e-01


.. parsed-literal::

     138	 1.4686220e+00	 1.4444245e-01	 1.5234220e+00	 1.5303362e-01	  1.3021305e+00 	 2.2689962e-01


.. parsed-literal::

     139	 1.4702249e+00	 1.4454721e-01	 1.5250352e+00	 1.5321534e-01	  1.3018880e+00 	 2.2310925e-01


.. parsed-literal::

     140	 1.4723124e+00	 1.4462869e-01	 1.5271565e+00	 1.5343318e-01	  1.3010645e+00 	 2.3051429e-01


.. parsed-literal::

     141	 1.4750266e+00	 1.4460308e-01	 1.5299380e+00	 1.5337256e-01	  1.2993593e+00 	 2.3001814e-01


.. parsed-literal::

     142	 1.4767071e+00	 1.4444748e-01	 1.5317102e+00	 1.5336017e-01	  1.2947832e+00 	 3.8172293e-01
     143	 1.4791081e+00	 1.4431390e-01	 1.5340962e+00	 1.5297572e-01	  1.2953597e+00 	 1.9442773e-01


.. parsed-literal::

     144	 1.4807434e+00	 1.4407647e-01	 1.5356951e+00	 1.5251168e-01	  1.2951920e+00 	 2.1238136e-01


.. parsed-literal::

     145	 1.4825817e+00	 1.4397288e-01	 1.5375246e+00	 1.5234810e-01	  1.2956528e+00 	 2.2020984e-01


.. parsed-literal::

     146	 1.4844043e+00	 1.4391814e-01	 1.5393538e+00	 1.5216629e-01	  1.2943364e+00 	 2.2946763e-01


.. parsed-literal::

     147	 1.4862019e+00	 1.4394446e-01	 1.5412014e+00	 1.5218119e-01	  1.2932934e+00 	 2.3607540e-01


.. parsed-literal::

     148	 1.4879136e+00	 1.4425063e-01	 1.5430513e+00	 1.5244980e-01	  1.2920039e+00 	 2.0682025e-01


.. parsed-literal::

     149	 1.4894282e+00	 1.4420089e-01	 1.5446110e+00	 1.5228765e-01	  1.2896763e+00 	 2.2917747e-01


.. parsed-literal::

     150	 1.4907169e+00	 1.4420731e-01	 1.5459417e+00	 1.5207263e-01	  1.2886260e+00 	 2.2692132e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 20s, sys: 1.24 s, total: 2min 21s
    Wall time: 35.5 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fd6fdc571d0>



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
    CPU times: user 1.8 s, sys: 63.9 ms, total: 1.87 s
    Wall time: 614 ms


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




.. image:: 06_GPz_files/06_GPz_16_1.png


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




.. image:: 06_GPz_files/06_GPz_19_1.png

