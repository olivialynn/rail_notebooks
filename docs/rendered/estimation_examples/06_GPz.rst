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
       1	-3.4238765e-01	 3.2025521e-01	-3.3263655e-01	 3.2259795e-01	[-3.3593930e-01]	 4.6169639e-01


.. parsed-literal::

       2	-2.7348523e-01	 3.1051942e-01	-2.5062598e-01	 3.1112852e-01	[-2.5247462e-01]	 2.2998595e-01


.. parsed-literal::

       3	-2.2904017e-01	 2.8993319e-01	-1.8843735e-01	 2.8935348e-01	[-1.8373873e-01]	 2.7883148e-01


.. parsed-literal::

       4	-1.9463624e-01	 2.6690939e-01	-1.5387220e-01	 2.6459900e-01	[-1.3803492e-01]	 2.0238996e-01


.. parsed-literal::

       5	-1.0759993e-01	 2.5814886e-01	-7.4252371e-02	 2.5677432e-01	[-6.6590423e-02]	 2.1160340e-01
       6	-7.2983197e-02	 2.5228990e-01	-4.3072975e-02	 2.4986672e-01	[-3.4013553e-02]	 1.9813991e-01


.. parsed-literal::

       7	-5.5356149e-02	 2.4960782e-01	-3.1473142e-02	 2.4754480e-01	[-2.3379510e-02]	 2.2012973e-01
       8	-4.1444799e-02	 2.4720817e-01	-2.1486677e-02	 2.4552496e-01	[-1.4684878e-02]	 1.8976808e-01


.. parsed-literal::

       9	-2.8753861e-02	 2.4486217e-01	-1.1554360e-02	 2.4256506e-01	[-1.9866334e-03]	 2.1721363e-01


.. parsed-literal::

      10	-1.9069308e-02	 2.4303851e-01	-3.8947784e-03	 2.4069102e-01	[ 5.5355878e-03]	 2.0356536e-01


.. parsed-literal::

      11	-1.2988464e-02	 2.4194547e-01	 1.1120653e-03	 2.3831953e-01	[ 1.6825346e-02]	 2.1904612e-01


.. parsed-literal::

      12	-1.0157033e-02	 2.4153304e-01	 3.7868259e-03	 2.3822850e-01	[ 1.7480808e-02]	 2.0897317e-01


.. parsed-literal::

      13	-7.2784529e-03	 2.4097637e-01	 6.5353051e-03	 2.3800841e-01	[ 1.8729203e-02]	 2.1517420e-01


.. parsed-literal::

      14	-1.2245400e-03	 2.3964887e-01	 1.3502181e-02	 2.3671826e-01	[ 2.5390347e-02]	 2.0990181e-01


.. parsed-literal::

      15	 5.8898083e-02	 2.2795287e-01	 7.6344478e-02	 2.2415152e-01	[ 8.7662519e-02]	 3.2190657e-01


.. parsed-literal::

      16	 6.4024267e-02	 2.2538956e-01	 8.3363981e-02	 2.2548515e-01	  8.6417557e-02 	 2.0435667e-01


.. parsed-literal::

      17	 1.5277625e-01	 2.2128632e-01	 1.7380081e-01	 2.2115929e-01	[ 1.7990042e-01]	 2.1946192e-01
      18	 2.5027898e-01	 2.1489446e-01	 2.8002213e-01	 2.1251818e-01	[ 3.0134020e-01]	 2.0223665e-01


.. parsed-literal::

      19	 3.0367752e-01	 2.1521065e-01	 3.3392813e-01	 2.1318287e-01	[ 3.5083844e-01]	 1.9100904e-01


.. parsed-literal::

      20	 3.3588612e-01	 2.1282785e-01	 3.6606310e-01	 2.1119639e-01	[ 3.7754945e-01]	 2.2137690e-01


.. parsed-literal::

      21	 3.8040241e-01	 2.0910464e-01	 4.1180497e-01	 2.1007754e-01	[ 4.1212919e-01]	 2.1949649e-01


.. parsed-literal::

      22	 4.6053619e-01	 2.0621948e-01	 4.9380047e-01	 2.1165892e-01	[ 4.8213687e-01]	 2.0390320e-01


.. parsed-literal::

      23	 5.4030023e-01	 2.1008762e-01	 5.7704814e-01	 2.1548876e-01	[ 5.7106267e-01]	 2.0756602e-01


.. parsed-literal::

      24	 5.8040770e-01	 2.0748387e-01	 6.1865598e-01	 2.1102239e-01	[ 6.2914645e-01]	 2.1058178e-01


.. parsed-literal::

      25	 6.1879923e-01	 2.0263745e-01	 6.5637292e-01	 2.0751328e-01	[ 6.6002132e-01]	 2.1824431e-01


.. parsed-literal::

      26	 6.7101495e-01	 1.9629421e-01	 7.0772859e-01	 2.0139011e-01	[ 7.0552622e-01]	 2.0783067e-01


.. parsed-literal::

      27	 7.0209144e-01	 2.0110070e-01	 7.3732434e-01	 2.0136715e-01	[ 7.4733023e-01]	 2.1699047e-01
      28	 7.4680933e-01	 1.9606969e-01	 7.8383696e-01	 1.9737840e-01	[ 7.8782182e-01]	 1.8770480e-01


.. parsed-literal::

      29	 7.7673133e-01	 1.9239229e-01	 8.1413468e-01	 1.9414824e-01	[ 8.2136093e-01]	 2.0439672e-01


.. parsed-literal::

      30	 7.9921898e-01	 1.9204603e-01	 8.3755494e-01	 1.9505018e-01	[ 8.3323041e-01]	 2.1843791e-01


.. parsed-literal::

      31	 8.1700246e-01	 1.9261885e-01	 8.5679568e-01	 1.9596427e-01	[ 8.4308830e-01]	 2.2110009e-01


.. parsed-literal::

      32	 8.4300447e-01	 1.8971875e-01	 8.8206097e-01	 1.9338775e-01	[ 8.6904431e-01]	 2.1129441e-01


.. parsed-literal::

      33	 8.6151086e-01	 1.8752686e-01	 9.0062395e-01	 1.9156339e-01	[ 8.9131317e-01]	 2.0872951e-01


.. parsed-literal::

      34	 8.8782874e-01	 1.8372736e-01	 9.2763488e-01	 1.8812658e-01	[ 9.2240877e-01]	 2.0652461e-01
      35	 9.0446598e-01	 1.8359483e-01	 9.4528206e-01	 1.8840227e-01	[ 9.3398618e-01]	 1.8801785e-01


.. parsed-literal::

      36	 9.2174223e-01	 1.8216116e-01	 9.6284253e-01	 1.8715337e-01	[ 9.4687497e-01]	 1.8408799e-01
      37	 9.3541537e-01	 1.8133295e-01	 9.7740528e-01	 1.8616817e-01	[ 9.5705091e-01]	 1.9097018e-01


.. parsed-literal::

      38	 9.5074416e-01	 1.8071026e-01	 9.9370936e-01	 1.8500466e-01	[ 9.6947905e-01]	 2.1968818e-01


.. parsed-literal::

      39	 9.6727442e-01	 1.8140735e-01	 1.0120683e+00	 1.8497622e-01	[ 9.7233022e-01]	 2.1818352e-01
      40	 9.8439845e-01	 1.8083289e-01	 1.0295036e+00	 1.8408389e-01	[ 9.8972111e-01]	 1.8073821e-01


.. parsed-literal::

      41	 9.9554473e-01	 1.8008600e-01	 1.0400907e+00	 1.8330379e-01	[ 1.0050583e+00]	 2.0265746e-01


.. parsed-literal::

      42	 1.0108006e+00	 1.7931561e-01	 1.0554071e+00	 1.8226687e-01	[ 1.0180705e+00]	 2.1027374e-01


.. parsed-literal::

      43	 1.0182312e+00	 1.7924311e-01	 1.0636577e+00	 1.8270087e-01	[ 1.0255345e+00]	 2.0969534e-01
      44	 1.0282795e+00	 1.7901845e-01	 1.0738948e+00	 1.8251061e-01	[ 1.0295273e+00]	 1.9797516e-01


.. parsed-literal::

      45	 1.0368832e+00	 1.7839722e-01	 1.0827213e+00	 1.8177585e-01	[ 1.0310871e+00]	 1.9669557e-01
      46	 1.0487383e+00	 1.7744913e-01	 1.0951223e+00	 1.8060493e-01	  1.0303572e+00 	 1.8236375e-01


.. parsed-literal::

      47	 1.0628662e+00	 1.7505683e-01	 1.1100446e+00	 1.7817888e-01	[ 1.0386812e+00]	 2.1102500e-01


.. parsed-literal::

      48	 1.0745316e+00	 1.7331976e-01	 1.1218325e+00	 1.7707308e-01	[ 1.0502332e+00]	 2.1794438e-01


.. parsed-literal::

      49	 1.0827801e+00	 1.7210885e-01	 1.1298840e+00	 1.7581567e-01	[ 1.0637154e+00]	 2.0862103e-01


.. parsed-literal::

      50	 1.0938851e+00	 1.7090310e-01	 1.1414142e+00	 1.7421936e-01	[ 1.0809442e+00]	 2.0435119e-01
      51	 1.1016820e+00	 1.7023409e-01	 1.1492945e+00	 1.7279166e-01	[ 1.0871630e+00]	 2.0550656e-01


.. parsed-literal::

      52	 1.1090709e+00	 1.6940022e-01	 1.1565439e+00	 1.7185141e-01	[ 1.0911773e+00]	 1.8498850e-01
      53	 1.1202935e+00	 1.6798295e-01	 1.1680334e+00	 1.7071554e-01	[ 1.0985897e+00]	 1.7783904e-01


.. parsed-literal::

      54	 1.1303681e+00	 1.6515640e-01	 1.1785119e+00	 1.6816210e-01	[ 1.1049046e+00]	 2.0463753e-01


.. parsed-literal::

      55	 1.1386593e+00	 1.6396994e-01	 1.1869899e+00	 1.6736333e-01	[ 1.1226324e+00]	 2.0809770e-01


.. parsed-literal::

      56	 1.1436574e+00	 1.6302539e-01	 1.1919381e+00	 1.6672627e-01	[ 1.1311540e+00]	 2.1056747e-01
      57	 1.1492490e+00	 1.6162842e-01	 1.1977368e+00	 1.6516360e-01	[ 1.1417775e+00]	 2.0044374e-01


.. parsed-literal::

      58	 1.1570026e+00	 1.5981076e-01	 1.2057335e+00	 1.6351641e-01	[ 1.1526544e+00]	 1.7843366e-01


.. parsed-literal::

      59	 1.1662280e+00	 1.5790532e-01	 1.2149738e+00	 1.6118729e-01	[ 1.1632465e+00]	 2.0614004e-01


.. parsed-literal::

      60	 1.1735092e+00	 1.5651854e-01	 1.2222259e+00	 1.5995206e-01	[ 1.1688248e+00]	 2.1874261e-01


.. parsed-literal::

      61	 1.1822615e+00	 1.5473081e-01	 1.2313344e+00	 1.5773142e-01	[ 1.1714946e+00]	 2.0751524e-01


.. parsed-literal::

      62	 1.1871894e+00	 1.5355387e-01	 1.2364646e+00	 1.5648889e-01	  1.1695056e+00 	 2.1752048e-01
      63	 1.1935092e+00	 1.5305508e-01	 1.2427631e+00	 1.5577616e-01	[ 1.1770833e+00]	 2.0662427e-01


.. parsed-literal::

      64	 1.2002088e+00	 1.5245963e-01	 1.2496438e+00	 1.5486344e-01	[ 1.1828992e+00]	 2.1073151e-01


.. parsed-literal::

      65	 1.2058446e+00	 1.5152439e-01	 1.2554314e+00	 1.5383468e-01	[ 1.1876453e+00]	 2.0926666e-01


.. parsed-literal::

      66	 1.2130656e+00	 1.5014894e-01	 1.2631592e+00	 1.5258932e-01	[ 1.1927506e+00]	 2.1067977e-01


.. parsed-literal::

      67	 1.2198261e+00	 1.4934795e-01	 1.2698643e+00	 1.5201810e-01	[ 1.2016544e+00]	 2.0907331e-01


.. parsed-literal::

      68	 1.2249569e+00	 1.4880593e-01	 1.2749802e+00	 1.5178413e-01	[ 1.2100686e+00]	 2.1681356e-01


.. parsed-literal::

      69	 1.2299153e+00	 1.4876023e-01	 1.2801916e+00	 1.5229967e-01	[ 1.2130829e+00]	 2.1086264e-01


.. parsed-literal::

      70	 1.2353596e+00	 1.4840501e-01	 1.2857547e+00	 1.5193959e-01	[ 1.2177080e+00]	 2.1374035e-01


.. parsed-literal::

      71	 1.2400670e+00	 1.4812183e-01	 1.2906273e+00	 1.5195180e-01	[ 1.2194479e+00]	 2.1088481e-01


.. parsed-literal::

      72	 1.2456717e+00	 1.4773817e-01	 1.2966169e+00	 1.5199174e-01	  1.2192567e+00 	 2.1222711e-01


.. parsed-literal::

      73	 1.2507640e+00	 1.4734063e-01	 1.3019885e+00	 1.5271997e-01	  1.2183370e+00 	 2.0213962e-01


.. parsed-literal::

      74	 1.2562832e+00	 1.4715806e-01	 1.3074842e+00	 1.5272806e-01	[ 1.2241915e+00]	 2.2393036e-01


.. parsed-literal::

      75	 1.2615876e+00	 1.4697823e-01	 1.3127389e+00	 1.5272788e-01	[ 1.2315854e+00]	 2.1092772e-01


.. parsed-literal::

      76	 1.2674948e+00	 1.4684996e-01	 1.3185986e+00	 1.5272900e-01	[ 1.2413509e+00]	 2.0667577e-01
      77	 1.2733844e+00	 1.4715269e-01	 1.3248307e+00	 1.5303466e-01	[ 1.2483613e+00]	 2.0112777e-01


.. parsed-literal::

      78	 1.2792892e+00	 1.4686334e-01	 1.3305968e+00	 1.5260861e-01	[ 1.2592268e+00]	 2.1482468e-01
      79	 1.2830413e+00	 1.4678366e-01	 1.3344283e+00	 1.5249720e-01	[ 1.2633758e+00]	 1.9840240e-01


.. parsed-literal::

      80	 1.2892837e+00	 1.4661412e-01	 1.3407885e+00	 1.5226183e-01	[ 1.2688411e+00]	 2.0611191e-01
      81	 1.2929142e+00	 1.4712305e-01	 1.3445833e+00	 1.5246771e-01	  1.2674196e+00 	 1.8914533e-01


.. parsed-literal::

      82	 1.2989966e+00	 1.4682909e-01	 1.3505674e+00	 1.5213723e-01	[ 1.2739652e+00]	 2.1448755e-01


.. parsed-literal::

      83	 1.3017940e+00	 1.4653839e-01	 1.3532957e+00	 1.5195587e-01	[ 1.2758522e+00]	 2.1002460e-01


.. parsed-literal::

      84	 1.3060948e+00	 1.4631170e-01	 1.3577051e+00	 1.5188825e-01	[ 1.2760161e+00]	 2.1762085e-01
      85	 1.3118421e+00	 1.4615045e-01	 1.3636359e+00	 1.5185904e-01	  1.2753276e+00 	 1.7510676e-01


.. parsed-literal::

      86	 1.3179933e+00	 1.4604963e-01	 1.3700409e+00	 1.5197956e-01	  1.2740579e+00 	 2.1880174e-01
      87	 1.3230605e+00	 1.4611723e-01	 1.3750906e+00	 1.5205090e-01	[ 1.2772707e+00]	 1.9085622e-01


.. parsed-literal::

      88	 1.3278120e+00	 1.4599735e-01	 1.3798811e+00	 1.5193064e-01	[ 1.2788058e+00]	 2.1666980e-01


.. parsed-literal::

      89	 1.3310392e+00	 1.4601885e-01	 1.3832718e+00	 1.5169319e-01	  1.2751807e+00 	 2.0808983e-01
      90	 1.3341940e+00	 1.4557402e-01	 1.3864240e+00	 1.5130680e-01	  1.2755799e+00 	 1.7416739e-01


.. parsed-literal::

      91	 1.3376994e+00	 1.4519960e-01	 1.3900407e+00	 1.5094900e-01	  1.2707493e+00 	 2.0599771e-01
      92	 1.3405671e+00	 1.4480583e-01	 1.3929718e+00	 1.5053117e-01	  1.2689039e+00 	 1.8666315e-01


.. parsed-literal::

      93	 1.3464069e+00	 1.4445062e-01	 1.3990977e+00	 1.4991720e-01	  1.2618038e+00 	 1.8801975e-01


.. parsed-literal::

      94	 1.3499103e+00	 1.4390968e-01	 1.4028451e+00	 1.4937751e-01	  1.2545226e+00 	 2.0755863e-01


.. parsed-literal::

      95	 1.3531965e+00	 1.4402164e-01	 1.4060460e+00	 1.4949506e-01	  1.2585125e+00 	 2.0748544e-01


.. parsed-literal::

      96	 1.3554097e+00	 1.4400274e-01	 1.4082119e+00	 1.4945390e-01	  1.2628283e+00 	 2.0738173e-01
      97	 1.3582784e+00	 1.4365041e-01	 1.4111269e+00	 1.4910755e-01	  1.2643279e+00 	 1.9417858e-01


.. parsed-literal::

      98	 1.3633603e+00	 1.4275772e-01	 1.4162316e+00	 1.4822357e-01	  1.2670842e+00 	 2.1569133e-01


.. parsed-literal::

      99	 1.3661320e+00	 1.4197970e-01	 1.4191430e+00	 1.4753364e-01	  1.2628349e+00 	 2.9745507e-01
     100	 1.3696687e+00	 1.4132902e-01	 1.4227108e+00	 1.4688397e-01	  1.2614305e+00 	 1.8293381e-01


.. parsed-literal::

     101	 1.3730555e+00	 1.4074128e-01	 1.4261739e+00	 1.4642729e-01	  1.2598035e+00 	 2.0967627e-01


.. parsed-literal::

     102	 1.3765560e+00	 1.4023796e-01	 1.4297907e+00	 1.4609723e-01	  1.2593614e+00 	 2.1888947e-01


.. parsed-literal::

     103	 1.3800570e+00	 1.3945671e-01	 1.4335609e+00	 1.4587780e-01	  1.2642074e+00 	 2.1839356e-01


.. parsed-literal::

     104	 1.3846136e+00	 1.3930975e-01	 1.4379648e+00	 1.4588656e-01	  1.2705413e+00 	 2.1120882e-01


.. parsed-literal::

     105	 1.3866966e+00	 1.3923768e-01	 1.4399578e+00	 1.4583477e-01	  1.2743703e+00 	 2.1887016e-01


.. parsed-literal::

     106	 1.3899885e+00	 1.3884490e-01	 1.4432448e+00	 1.4565936e-01	[ 1.2796136e+00]	 2.0966697e-01


.. parsed-literal::

     107	 1.3922700e+00	 1.3835435e-01	 1.4456247e+00	 1.4515135e-01	  1.2781661e+00 	 2.9753995e-01


.. parsed-literal::

     108	 1.3958712e+00	 1.3783698e-01	 1.4492578e+00	 1.4489558e-01	[ 1.2808788e+00]	 2.1111798e-01
     109	 1.3999818e+00	 1.3693625e-01	 1.4535437e+00	 1.4434898e-01	  1.2799522e+00 	 1.8300509e-01


.. parsed-literal::

     110	 1.4032009e+00	 1.3651075e-01	 1.4568120e+00	 1.4399645e-01	  1.2802339e+00 	 2.0336342e-01
     111	 1.4055530e+00	 1.3602569e-01	 1.4592257e+00	 1.4358994e-01	  1.2808072e+00 	 1.9898129e-01


.. parsed-literal::

     112	 1.4076710e+00	 1.3600234e-01	 1.4612620e+00	 1.4361750e-01	[ 1.2840861e+00]	 1.7641187e-01
     113	 1.4102000e+00	 1.3597746e-01	 1.4637174e+00	 1.4366082e-01	[ 1.2853152e+00]	 1.8334103e-01


.. parsed-literal::

     114	 1.4137075e+00	 1.3581361e-01	 1.4672384e+00	 1.4366682e-01	  1.2850174e+00 	 2.0867991e-01
     115	 1.4146728e+00	 1.3559949e-01	 1.4685076e+00	 1.4394943e-01	  1.2689151e+00 	 1.9618607e-01


.. parsed-literal::

     116	 1.4194347e+00	 1.3536282e-01	 1.4731113e+00	 1.4363719e-01	  1.2795993e+00 	 2.1353936e-01


.. parsed-literal::

     117	 1.4210802e+00	 1.3514069e-01	 1.4747771e+00	 1.4346038e-01	  1.2826546e+00 	 2.0453763e-01


.. parsed-literal::

     118	 1.4235628e+00	 1.3476546e-01	 1.4773267e+00	 1.4325661e-01	[ 1.2860530e+00]	 2.0547581e-01


.. parsed-literal::

     119	 1.4268355e+00	 1.3424956e-01	 1.4806642e+00	 1.4305827e-01	[ 1.2906501e+00]	 2.1064115e-01
     120	 1.4295125e+00	 1.3353290e-01	 1.4834597e+00	 1.4278641e-01	[ 1.2999766e+00]	 2.0805717e-01


.. parsed-literal::

     121	 1.4327865e+00	 1.3360490e-01	 1.4865670e+00	 1.4290736e-01	[ 1.3011809e+00]	 2.0037055e-01


.. parsed-literal::

     122	 1.4348618e+00	 1.3363295e-01	 1.4885705e+00	 1.4297033e-01	[ 1.3026515e+00]	 2.0796561e-01


.. parsed-literal::

     123	 1.4365081e+00	 1.3366130e-01	 1.4902705e+00	 1.4295585e-01	[ 1.3039977e+00]	 2.0961285e-01


.. parsed-literal::

     124	 1.4386643e+00	 1.3351808e-01	 1.4924472e+00	 1.4287065e-01	[ 1.3050110e+00]	 2.0674443e-01


.. parsed-literal::

     125	 1.4405749e+00	 1.3330982e-01	 1.4944577e+00	 1.4273649e-01	  1.3041566e+00 	 2.0192122e-01


.. parsed-literal::

     126	 1.4423738e+00	 1.3313786e-01	 1.4963663e+00	 1.4262135e-01	  1.3018174e+00 	 2.0563722e-01
     127	 1.4441561e+00	 1.3273153e-01	 1.4983378e+00	 1.4230390e-01	  1.2949871e+00 	 1.7750764e-01


.. parsed-literal::

     128	 1.4469234e+00	 1.3265190e-01	 1.5010890e+00	 1.4217936e-01	  1.2945455e+00 	 2.0662451e-01


.. parsed-literal::

     129	 1.4485799e+00	 1.3256527e-01	 1.5026972e+00	 1.4205745e-01	  1.2953892e+00 	 2.0224929e-01


.. parsed-literal::

     130	 1.4506157e+00	 1.3239611e-01	 1.5047070e+00	 1.4181738e-01	  1.2950517e+00 	 2.0550060e-01
     131	 1.4537685e+00	 1.3211953e-01	 1.5079123e+00	 1.4144419e-01	  1.2915523e+00 	 2.0044398e-01


.. parsed-literal::

     132	 1.4555985e+00	 1.3195752e-01	 1.5098179e+00	 1.4117678e-01	  1.2897642e+00 	 3.2865405e-01


.. parsed-literal::

     133	 1.4578738e+00	 1.3174642e-01	 1.5121934e+00	 1.4092350e-01	  1.2828030e+00 	 2.0371389e-01


.. parsed-literal::

     134	 1.4593686e+00	 1.3163880e-01	 1.5137090e+00	 1.4070128e-01	  1.2853039e+00 	 2.0621037e-01
     135	 1.4612760e+00	 1.3151208e-01	 1.5156690e+00	 1.4056276e-01	  1.2804313e+00 	 2.0846343e-01


.. parsed-literal::

     136	 1.4636489e+00	 1.3117730e-01	 1.5181040e+00	 1.4007378e-01	  1.2760881e+00 	 2.0795774e-01


.. parsed-literal::

     137	 1.4662111e+00	 1.3093036e-01	 1.5206795e+00	 1.3985222e-01	  1.2686803e+00 	 2.1388292e-01


.. parsed-literal::

     138	 1.4677265e+00	 1.3067424e-01	 1.5222887e+00	 1.3958041e-01	  1.2592477e+00 	 2.1792579e-01
     139	 1.4692616e+00	 1.3054386e-01	 1.5237150e+00	 1.3945450e-01	  1.2633740e+00 	 1.8711162e-01


.. parsed-literal::

     140	 1.4707439e+00	 1.3041632e-01	 1.5251332e+00	 1.3936179e-01	  1.2672608e+00 	 2.0370245e-01


.. parsed-literal::

     141	 1.4723370e+00	 1.3027503e-01	 1.5267377e+00	 1.3929429e-01	  1.2672064e+00 	 2.1284318e-01


.. parsed-literal::

     142	 1.4734251e+00	 1.3015142e-01	 1.5279334e+00	 1.3936151e-01	  1.2705065e+00 	 2.0900297e-01
     143	 1.4754309e+00	 1.3009393e-01	 1.5299100e+00	 1.3934198e-01	  1.2658145e+00 	 1.8807101e-01


.. parsed-literal::

     144	 1.4763100e+00	 1.3010072e-01	 1.5307996e+00	 1.3937658e-01	  1.2630062e+00 	 2.2057056e-01
     145	 1.4775126e+00	 1.3005788e-01	 1.5320532e+00	 1.3940652e-01	  1.2587592e+00 	 1.9284177e-01


.. parsed-literal::

     146	 1.4794759e+00	 1.2992335e-01	 1.5341198e+00	 1.3940125e-01	  1.2505442e+00 	 2.1621633e-01


.. parsed-literal::

     147	 1.4809672e+00	 1.2983535e-01	 1.5357110e+00	 1.3947676e-01	  1.2503257e+00 	 3.3142328e-01


.. parsed-literal::

     148	 1.4831974e+00	 1.2970683e-01	 1.5380169e+00	 1.3953826e-01	  1.2409439e+00 	 2.1531963e-01


.. parsed-literal::

     149	 1.4849005e+00	 1.2959792e-01	 1.5397594e+00	 1.3945282e-01	  1.2424848e+00 	 2.1371937e-01
     150	 1.4863325e+00	 1.2965375e-01	 1.5412242e+00	 1.3961030e-01	  1.2402900e+00 	 1.9189620e-01


.. parsed-literal::

    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 6s, sys: 1.07 s, total: 2min 7s
    Wall time: 31.9 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fb6c8470f40>



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
    CPU times: user 2.08 s, sys: 47.9 ms, total: 2.13 s
    Wall time: 639 ms


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

