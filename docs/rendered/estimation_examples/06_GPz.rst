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
       1	-3.4678206e-01	 3.2145579e-01	-3.3711375e-01	 3.1747608e-01	[-3.2943165e-01]	 4.6379471e-01


.. parsed-literal::

       2	-2.7534836e-01	 3.1072326e-01	-2.5123526e-01	 3.0614721e-01	[-2.3739954e-01]	 2.3081398e-01


.. parsed-literal::

       3	-2.3108485e-01	 2.9000630e-01	-1.8887375e-01	 2.8439681e-01	[-1.6306458e-01]	 2.9323792e-01


.. parsed-literal::

       4	-1.9535616e-01	 2.6666035e-01	-1.5442387e-01	 2.5837529e-01	[-1.0768920e-01]	 2.0435095e-01


.. parsed-literal::

       5	-1.0585757e-01	 2.5750529e-01	-7.1889344e-02	 2.5293963e-01	[-4.8663859e-02]	 2.0139790e-01
       6	-7.2081052e-02	 2.5217250e-01	-4.2013909e-02	 2.4799895e-01	[-2.6773006e-02]	 1.9347024e-01


.. parsed-literal::

       7	-5.4965905e-02	 2.4945449e-01	-3.0855215e-02	 2.4584155e-01	[-1.6528530e-02]	 2.1448350e-01


.. parsed-literal::

       8	-4.2697163e-02	 2.4742310e-01	-2.2445027e-02	 2.4411375e-01	[-8.4846027e-03]	 2.0948792e-01
       9	-2.9729561e-02	 2.4501190e-01	-1.2278465e-02	 2.4231552e-01	[ 3.8980922e-04]	 1.8060350e-01


.. parsed-literal::

      10	-1.7826249e-02	 2.4265018e-01	-2.4402359e-03	 2.4115000e-01	[ 7.1773009e-03]	 1.9640899e-01


.. parsed-literal::

      11	-1.4649007e-02	 2.4232463e-01	-4.8323745e-04	 2.4070260e-01	[ 8.3905960e-03]	 2.1890521e-01


.. parsed-literal::

      12	-1.0550714e-02	 2.4157384e-01	 3.3176667e-03	 2.4001957e-01	[ 1.2434756e-02]	 2.1618223e-01


.. parsed-literal::

      13	-7.5672199e-03	 2.4094729e-01	 6.2619863e-03	 2.3909927e-01	[ 1.6187390e-02]	 2.0658422e-01
      14	-4.0726875e-03	 2.4020909e-01	 1.0130412e-02	 2.3806105e-01	[ 2.1003728e-02]	 1.8482065e-01


.. parsed-literal::

      15	 1.1174130e-01	 2.2670416e-01	 1.3270306e-01	 2.2185619e-01	[ 1.4670775e-01]	 3.1502628e-01


.. parsed-literal::

      16	 1.4740087e-01	 2.2452208e-01	 1.7093757e-01	 2.1776938e-01	[ 1.8525122e-01]	 3.2640433e-01


.. parsed-literal::

      17	 1.9843241e-01	 2.2015509e-01	 2.2262159e-01	 2.1403778e-01	[ 2.3551820e-01]	 2.1391726e-01


.. parsed-literal::

      18	 2.9175643e-01	 2.1616296e-01	 3.2054091e-01	 2.1055087e-01	[ 3.3270785e-01]	 2.1085620e-01


.. parsed-literal::

      19	 3.1850868e-01	 2.1538965e-01	 3.4971759e-01	 2.0928468e-01	[ 3.6447209e-01]	 2.0794511e-01
      20	 3.6072446e-01	 2.1221135e-01	 3.9284338e-01	 2.0505092e-01	[ 4.0888160e-01]	 1.8483448e-01


.. parsed-literal::

      21	 3.9888834e-01	 2.1137839e-01	 4.3105679e-01	 2.0378498e-01	[ 4.4555094e-01]	 1.9424033e-01
      22	 4.8263032e-01	 2.1040815e-01	 5.1575885e-01	 2.0531939e-01	[ 5.1730211e-01]	 1.7285681e-01


.. parsed-literal::

      23	 5.5607883e-01	 2.0882072e-01	 5.9170884e-01	 2.0600707e-01	[ 6.0273975e-01]	 2.0032287e-01
      24	 5.9384856e-01	 2.0716111e-01	 6.3362519e-01	 2.0226414e-01	[ 6.6425307e-01]	 1.9834805e-01


.. parsed-literal::

      25	 6.2712391e-01	 2.0469501e-01	 6.6469227e-01	 2.0084440e-01	[ 6.8490120e-01]	 2.0612979e-01


.. parsed-literal::

      26	 6.5477237e-01	 2.0323252e-01	 6.9134121e-01	 2.0014371e-01	[ 7.1002843e-01]	 2.1192241e-01
      27	 6.8846294e-01	 2.0219410e-01	 7.2351528e-01	 1.9717657e-01	[ 7.3493885e-01]	 1.8783379e-01


.. parsed-literal::

      28	 7.1089213e-01	 2.0515069e-01	 7.4675618e-01	 1.9961020e-01	[ 7.5152450e-01]	 2.0722294e-01


.. parsed-literal::

      29	 7.3345763e-01	 2.0218582e-01	 7.7158227e-01	 1.9605927e-01	[ 7.8444268e-01]	 2.1175504e-01


.. parsed-literal::

      30	 7.5769340e-01	 2.0153789e-01	 7.9594886e-01	 1.9610699e-01	[ 8.0572970e-01]	 2.0505190e-01
      31	 7.7428834e-01	 2.0500303e-01	 8.1253466e-01	 1.9943909e-01	[ 8.1840875e-01]	 1.9630671e-01


.. parsed-literal::

      32	 8.0063591e-01	 2.0425825e-01	 8.3918228e-01	 2.0009282e-01	[ 8.4148269e-01]	 2.0880771e-01


.. parsed-literal::

      33	 8.3184010e-01	 2.0349924e-01	 8.7166463e-01	 1.9880446e-01	[ 8.7454262e-01]	 2.2082019e-01
      34	 8.5630333e-01	 2.0047979e-01	 8.9773977e-01	 1.9681769e-01	[ 8.9101228e-01]	 2.0500851e-01


.. parsed-literal::

      35	 8.7578595e-01	 1.9888572e-01	 9.1799811e-01	 1.9481637e-01	[ 9.0705631e-01]	 1.8112779e-01


.. parsed-literal::

      36	 8.9004017e-01	 1.9675127e-01	 9.3203692e-01	 1.9327016e-01	[ 9.1781790e-01]	 2.0525694e-01
      37	 9.1066895e-01	 1.9247361e-01	 9.5259520e-01	 1.9070118e-01	[ 9.3279691e-01]	 1.7417264e-01


.. parsed-literal::

      38	 9.2906874e-01	 1.9000359e-01	 9.7115112e-01	 1.8773116e-01	[ 9.4733987e-01]	 1.8419814e-01
      39	 9.5321116e-01	 1.8746195e-01	 9.9597700e-01	 1.8479154e-01	[ 9.7211536e-01]	 1.9975281e-01


.. parsed-literal::

      40	 9.6970627e-01	 1.8644103e-01	 1.0138199e+00	 1.8349181e-01	[ 9.8828280e-01]	 1.8629599e-01
      41	 9.8605743e-01	 1.8531571e-01	 1.0316567e+00	 1.8018534e-01	[ 1.0045514e+00]	 1.9626665e-01


.. parsed-literal::

      42	 1.0005394e+00	 1.8399355e-01	 1.0465222e+00	 1.7917751e-01	[ 1.0161322e+00]	 2.1760392e-01


.. parsed-literal::

      43	 1.0135280e+00	 1.8217611e-01	 1.0598267e+00	 1.7797979e-01	[ 1.0275505e+00]	 2.1864939e-01


.. parsed-literal::

      44	 1.0298339e+00	 1.8004700e-01	 1.0769150e+00	 1.7573682e-01	[ 1.0320596e+00]	 2.1154261e-01


.. parsed-literal::

      45	 1.0447251e+00	 1.7666776e-01	 1.0919170e+00	 1.7458585e-01	[ 1.0455193e+00]	 2.1289682e-01
      46	 1.0558534e+00	 1.7549969e-01	 1.1026986e+00	 1.7367940e-01	[ 1.0571638e+00]	 1.9926977e-01


.. parsed-literal::

      47	 1.0698694e+00	 1.7260408e-01	 1.1165584e+00	 1.7122600e-01	[ 1.0703128e+00]	 2.1547532e-01


.. parsed-literal::

      48	 1.0855563e+00	 1.6807197e-01	 1.1321889e+00	 1.6594432e-01	[ 1.0927849e+00]	 2.2022891e-01


.. parsed-literal::

      49	 1.0994558e+00	 1.6277425e-01	 1.1465195e+00	 1.5955221e-01	[ 1.1127982e+00]	 2.1480155e-01


.. parsed-literal::

      50	 1.1122166e+00	 1.6050660e-01	 1.1593593e+00	 1.5699461e-01	[ 1.1165184e+00]	 2.0327663e-01
      51	 1.1242206e+00	 1.5772429e-01	 1.1719185e+00	 1.5351246e-01	[ 1.1203913e+00]	 1.7649460e-01


.. parsed-literal::

      52	 1.1351477e+00	 1.5451767e-01	 1.1841121e+00	 1.5071374e-01	  1.1115862e+00 	 2.0356202e-01


.. parsed-literal::

      53	 1.1478016e+00	 1.5206869e-01	 1.1969228e+00	 1.4801838e-01	[ 1.1205635e+00]	 2.1643424e-01
      54	 1.1620691e+00	 1.4928362e-01	 1.2117031e+00	 1.4572503e-01	[ 1.1307205e+00]	 1.8566513e-01


.. parsed-literal::

      55	 1.1742883e+00	 1.4772991e-01	 1.2243576e+00	 1.4413670e-01	[ 1.1362826e+00]	 2.0168996e-01


.. parsed-literal::

      56	 1.1895726e+00	 1.4554369e-01	 1.2400539e+00	 1.4243003e-01	  1.1323192e+00 	 2.1023226e-01


.. parsed-literal::

      57	 1.2015451e+00	 1.4385503e-01	 1.2518827e+00	 1.3919173e-01	[ 1.1448317e+00]	 2.1124005e-01


.. parsed-literal::

      58	 1.2102189e+00	 1.4320820e-01	 1.2603127e+00	 1.3781838e-01	[ 1.1496276e+00]	 2.1949911e-01


.. parsed-literal::

      59	 1.2206800e+00	 1.4149901e-01	 1.2709999e+00	 1.3704850e-01	  1.1463377e+00 	 2.1015811e-01


.. parsed-literal::

      60	 1.2313309e+00	 1.4102304e-01	 1.2815235e+00	 1.3717975e-01	  1.1493083e+00 	 2.0936370e-01


.. parsed-literal::

      61	 1.2446946e+00	 1.4038938e-01	 1.2950139e+00	 1.3785379e-01	  1.1475927e+00 	 2.1503091e-01


.. parsed-literal::

      62	 1.2571347e+00	 1.3896511e-01	 1.3076655e+00	 1.3798634e-01	  1.1371877e+00 	 2.1832371e-01


.. parsed-literal::

      63	 1.2676770e+00	 1.3841332e-01	 1.3181744e+00	 1.3898275e-01	  1.1265380e+00 	 2.0952034e-01


.. parsed-literal::

      64	 1.2766326e+00	 1.3743221e-01	 1.3271227e+00	 1.3780960e-01	  1.1277401e+00 	 2.0517683e-01


.. parsed-literal::

      65	 1.2894614e+00	 1.3600600e-01	 1.3402180e+00	 1.3637817e-01	  1.1174651e+00 	 2.0591068e-01
      66	 1.2997489e+00	 1.3512435e-01	 1.3503228e+00	 1.3491592e-01	  1.1243783e+00 	 1.7558765e-01


.. parsed-literal::

      67	 1.3107150e+00	 1.3446210e-01	 1.3614088e+00	 1.3495042e-01	  1.1237851e+00 	 2.1339297e-01
      68	 1.3214357e+00	 1.3418346e-01	 1.3725591e+00	 1.3454010e-01	  1.1331949e+00 	 1.9699430e-01


.. parsed-literal::

      69	 1.3261298e+00	 1.3372084e-01	 1.3776786e+00	 1.3459427e-01	  1.1291650e+00 	 2.0685673e-01


.. parsed-literal::

      70	 1.3356449e+00	 1.3344282e-01	 1.3867597e+00	 1.3382121e-01	  1.1444713e+00 	 2.2038746e-01
      71	 1.3411625e+00	 1.3324650e-01	 1.3925176e+00	 1.3404681e-01	  1.1481898e+00 	 1.8385148e-01


.. parsed-literal::

      72	 1.3470094e+00	 1.3294712e-01	 1.3986299e+00	 1.3380237e-01	[ 1.1498762e+00]	 1.9844198e-01
      73	 1.3568718e+00	 1.3268172e-01	 1.4089752e+00	 1.3341973e-01	[ 1.1665642e+00]	 2.0039034e-01


.. parsed-literal::

      74	 1.3619317e+00	 1.3247911e-01	 1.4143440e+00	 1.3304646e-01	[ 1.1793376e+00]	 1.9030857e-01


.. parsed-literal::

      75	 1.3684463e+00	 1.3236576e-01	 1.4205455e+00	 1.3290099e-01	[ 1.1953784e+00]	 2.1660042e-01


.. parsed-literal::

      76	 1.3721053e+00	 1.3233205e-01	 1.4241858e+00	 1.3296159e-01	[ 1.2074305e+00]	 2.1958733e-01


.. parsed-literal::

      77	 1.3784299e+00	 1.3216131e-01	 1.4305163e+00	 1.3306951e-01	[ 1.2313565e+00]	 2.1714997e-01


.. parsed-literal::

      78	 1.3852263e+00	 1.3202377e-01	 1.4372932e+00	 1.3369743e-01	[ 1.2646304e+00]	 2.0831227e-01


.. parsed-literal::

      79	 1.3925442e+00	 1.3145476e-01	 1.4446329e+00	 1.3323804e-01	[ 1.2861072e+00]	 2.0954275e-01


.. parsed-literal::

      80	 1.3979789e+00	 1.3109343e-01	 1.4501125e+00	 1.3295681e-01	[ 1.2945567e+00]	 2.1883774e-01
      81	 1.4047935e+00	 1.3061101e-01	 1.4571498e+00	 1.3329125e-01	  1.2928220e+00 	 1.9578171e-01


.. parsed-literal::

      82	 1.4099732e+00	 1.2951117e-01	 1.4623668e+00	 1.3209425e-01	  1.2920463e+00 	 1.8741798e-01


.. parsed-literal::

      83	 1.4149828e+00	 1.2921372e-01	 1.4674036e+00	 1.3240616e-01	  1.2891609e+00 	 2.1298456e-01
      84	 1.4228647e+00	 1.2873236e-01	 1.4758646e+00	 1.3332490e-01	  1.2731415e+00 	 1.7650199e-01


.. parsed-literal::

      85	 1.4268615e+00	 1.2825496e-01	 1.4798422e+00	 1.3315100e-01	  1.2751982e+00 	 2.0892119e-01
      86	 1.4306572e+00	 1.2813437e-01	 1.4835271e+00	 1.3273559e-01	  1.2834479e+00 	 1.9936514e-01


.. parsed-literal::

      87	 1.4360147e+00	 1.2773407e-01	 1.4890692e+00	 1.3190465e-01	  1.2864717e+00 	 1.7233610e-01
      88	 1.4398425e+00	 1.2740999e-01	 1.4929739e+00	 1.3135770e-01	  1.2873431e+00 	 1.9140482e-01


.. parsed-literal::

      89	 1.4433275e+00	 1.2679853e-01	 1.4969062e+00	 1.3055756e-01	  1.2839464e+00 	 2.1107531e-01
      90	 1.4503060e+00	 1.2669261e-01	 1.5036205e+00	 1.3060951e-01	  1.2870091e+00 	 1.8083072e-01


.. parsed-literal::

      91	 1.4523993e+00	 1.2662642e-01	 1.5056383e+00	 1.3092238e-01	  1.2871120e+00 	 2.0315242e-01


.. parsed-literal::

      92	 1.4568540e+00	 1.2639477e-01	 1.5101575e+00	 1.3119000e-01	  1.2829848e+00 	 2.0698905e-01


.. parsed-literal::

      93	 1.4589381e+00	 1.2637255e-01	 1.5122552e+00	 1.3210721e-01	  1.2747578e+00 	 2.1401429e-01
      94	 1.4637047e+00	 1.2611960e-01	 1.5168665e+00	 1.3128285e-01	  1.2823562e+00 	 2.0280719e-01


.. parsed-literal::

      95	 1.4662251e+00	 1.2597267e-01	 1.5193914e+00	 1.3076989e-01	  1.2839235e+00 	 2.1606970e-01
      96	 1.4690643e+00	 1.2585179e-01	 1.5222547e+00	 1.3059258e-01	  1.2834190e+00 	 1.8601441e-01


.. parsed-literal::

      97	 1.4743505e+00	 1.2574242e-01	 1.5276524e+00	 1.3073538e-01	  1.2813287e+00 	 2.0565486e-01


.. parsed-literal::

      98	 1.4774501e+00	 1.2567751e-01	 1.5307616e+00	 1.3145062e-01	  1.2745209e+00 	 3.2103872e-01
      99	 1.4815903e+00	 1.2560407e-01	 1.5349731e+00	 1.3208141e-01	  1.2718045e+00 	 1.7107630e-01


.. parsed-literal::

     100	 1.4840881e+00	 1.2546208e-01	 1.5374799e+00	 1.3199333e-01	  1.2716199e+00 	 2.1318007e-01


.. parsed-literal::

     101	 1.4870338e+00	 1.2528314e-01	 1.5405111e+00	 1.3246816e-01	  1.2635244e+00 	 2.0980024e-01
     102	 1.4902581e+00	 1.2491150e-01	 1.5439140e+00	 1.3254547e-01	  1.2518164e+00 	 2.0644617e-01


.. parsed-literal::

     103	 1.4932428e+00	 1.2464839e-01	 1.5470011e+00	 1.3261472e-01	  1.2437084e+00 	 1.9408488e-01


.. parsed-literal::

     104	 1.4958469e+00	 1.2455410e-01	 1.5496701e+00	 1.3303359e-01	  1.2357355e+00 	 2.0938349e-01


.. parsed-literal::

     105	 1.4984115e+00	 1.2433265e-01	 1.5522199e+00	 1.3280466e-01	  1.2349027e+00 	 2.1936893e-01


.. parsed-literal::

     106	 1.5024727e+00	 1.2402154e-01	 1.5563775e+00	 1.3275427e-01	  1.2260579e+00 	 2.1408296e-01


.. parsed-literal::

     107	 1.5047067e+00	 1.2349590e-01	 1.5586243e+00	 1.3179494e-01	  1.2216266e+00 	 2.0783448e-01


.. parsed-literal::

     108	 1.5067054e+00	 1.2346610e-01	 1.5605533e+00	 1.3170520e-01	  1.2253695e+00 	 2.0376205e-01


.. parsed-literal::

     109	 1.5086129e+00	 1.2335720e-01	 1.5624870e+00	 1.3165783e-01	  1.2238171e+00 	 2.1865582e-01


.. parsed-literal::

     110	 1.5105426e+00	 1.2320234e-01	 1.5644395e+00	 1.3135746e-01	  1.2237617e+00 	 2.1352839e-01


.. parsed-literal::

     111	 1.5123397e+00	 1.2293218e-01	 1.5663332e+00	 1.3108040e-01	  1.2225655e+00 	 2.1213317e-01


.. parsed-literal::

     112	 1.5143950e+00	 1.2284430e-01	 1.5683265e+00	 1.3095683e-01	  1.2263628e+00 	 2.1571302e-01


.. parsed-literal::

     113	 1.5158318e+00	 1.2276561e-01	 1.5697463e+00	 1.3100314e-01	  1.2285652e+00 	 2.1704674e-01


.. parsed-literal::

     114	 1.5182635e+00	 1.2261996e-01	 1.5722246e+00	 1.3102946e-01	  1.2318478e+00 	 2.2128963e-01
     115	 1.5192690e+00	 1.2244632e-01	 1.5734714e+00	 1.3097591e-01	  1.2257582e+00 	 1.9980073e-01


.. parsed-literal::

     116	 1.5224805e+00	 1.2240447e-01	 1.5765710e+00	 1.3085547e-01	  1.2311429e+00 	 2.1118379e-01


.. parsed-literal::

     117	 1.5239118e+00	 1.2239202e-01	 1.5780180e+00	 1.3073213e-01	  1.2303669e+00 	 2.0455337e-01


.. parsed-literal::

     118	 1.5260549e+00	 1.2239795e-01	 1.5802537e+00	 1.3067533e-01	  1.2279576e+00 	 2.1173048e-01


.. parsed-literal::

     119	 1.5273205e+00	 1.2238487e-01	 1.5816236e+00	 1.3068893e-01	  1.2218042e+00 	 3.2312965e-01


.. parsed-literal::

     120	 1.5292436e+00	 1.2236626e-01	 1.5836230e+00	 1.3067208e-01	  1.2226676e+00 	 2.0372510e-01


.. parsed-literal::

     121	 1.5309425e+00	 1.2232005e-01	 1.5853883e+00	 1.3085415e-01	  1.2266820e+00 	 2.0381832e-01


.. parsed-literal::

     122	 1.5324973e+00	 1.2214907e-01	 1.5870088e+00	 1.3077155e-01	  1.2311191e+00 	 2.0865178e-01
     123	 1.5340437e+00	 1.2205447e-01	 1.5886258e+00	 1.3093556e-01	  1.2346770e+00 	 1.7823410e-01


.. parsed-literal::

     124	 1.5353937e+00	 1.2194830e-01	 1.5899716e+00	 1.3088295e-01	  1.2350042e+00 	 2.1281624e-01


.. parsed-literal::

     125	 1.5373478e+00	 1.2170315e-01	 1.5920467e+00	 1.3060170e-01	  1.2296708e+00 	 2.0912933e-01


.. parsed-literal::

     126	 1.5386665e+00	 1.2159247e-01	 1.5934925e+00	 1.3061543e-01	  1.2262032e+00 	 2.1183801e-01
     127	 1.5398699e+00	 1.2156275e-01	 1.5946557e+00	 1.3054416e-01	  1.2263954e+00 	 2.0421720e-01


.. parsed-literal::

     128	 1.5417893e+00	 1.2143675e-01	 1.5966467e+00	 1.3039767e-01	  1.2267214e+00 	 1.9111013e-01


.. parsed-literal::

     129	 1.5428580e+00	 1.2131256e-01	 1.5977854e+00	 1.3037863e-01	  1.2280730e+00 	 2.1165872e-01


.. parsed-literal::

     130	 1.5442627e+00	 1.2117441e-01	 1.5991843e+00	 1.3023728e-01	  1.2327098e+00 	 2.1122599e-01


.. parsed-literal::

     131	 1.5459614e+00	 1.2093742e-01	 1.6008949e+00	 1.3006917e-01	  1.2403533e+00 	 2.1801376e-01


.. parsed-literal::

     132	 1.5469274e+00	 1.2084737e-01	 1.6018307e+00	 1.3000565e-01	  1.2440707e+00 	 2.1207118e-01


.. parsed-literal::

     133	 1.5478286e+00	 1.2068636e-01	 1.6027645e+00	 1.2996124e-01	  1.2413702e+00 	 2.0295262e-01
     134	 1.5496098e+00	 1.2063247e-01	 1.6044751e+00	 1.2992227e-01	  1.2467602e+00 	 1.9048214e-01


.. parsed-literal::

     135	 1.5500906e+00	 1.2065783e-01	 1.6049543e+00	 1.2999702e-01	  1.2443382e+00 	 2.1762633e-01
     136	 1.5515490e+00	 1.2062302e-01	 1.6064924e+00	 1.3009143e-01	  1.2368628e+00 	 1.9159842e-01


.. parsed-literal::

     137	 1.5525767e+00	 1.2054326e-01	 1.6077009e+00	 1.3071235e-01	  1.2247350e+00 	 2.0337486e-01


.. parsed-literal::

     138	 1.5540207e+00	 1.2042131e-01	 1.6091165e+00	 1.3048737e-01	  1.2266903e+00 	 2.1963501e-01
     139	 1.5551278e+00	 1.2024949e-01	 1.6102448e+00	 1.3038809e-01	  1.2295978e+00 	 2.0704675e-01


.. parsed-literal::

     140	 1.5560059e+00	 1.2013964e-01	 1.6111346e+00	 1.3043684e-01	  1.2290460e+00 	 2.0695066e-01


.. parsed-literal::

     141	 1.5572478e+00	 1.1995911e-01	 1.6124198e+00	 1.3071724e-01	  1.2289132e+00 	 2.2811294e-01


.. parsed-literal::

     142	 1.5584360e+00	 1.1985109e-01	 1.6136057e+00	 1.3081081e-01	  1.2251373e+00 	 2.1814537e-01
     143	 1.5592828e+00	 1.1987075e-01	 1.6144292e+00	 1.3085150e-01	  1.2244666e+00 	 1.9553208e-01


.. parsed-literal::

     144	 1.5608669e+00	 1.1987866e-01	 1.6160108e+00	 1.3084183e-01	  1.2243982e+00 	 2.0138717e-01
     145	 1.5617830e+00	 1.1985554e-01	 1.6169732e+00	 1.3082196e-01	  1.2222744e+00 	 1.9178510e-01


.. parsed-literal::

     146	 1.5629419e+00	 1.1981874e-01	 1.6181404e+00	 1.3064316e-01	  1.2239702e+00 	 1.7480230e-01


.. parsed-literal::

     147	 1.5647267e+00	 1.1973077e-01	 1.6200028e+00	 1.3037397e-01	  1.2275755e+00 	 2.0280051e-01
     148	 1.5657603e+00	 1.1970505e-01	 1.6211045e+00	 1.3023677e-01	  1.2235064e+00 	 2.0638800e-01


.. parsed-literal::

     149	 1.5668305e+00	 1.1964623e-01	 1.6221899e+00	 1.3019333e-01	  1.2230605e+00 	 2.0419669e-01
     150	 1.5683255e+00	 1.1953073e-01	 1.6237557e+00	 1.3036462e-01	  1.2165815e+00 	 1.8132925e-01


.. parsed-literal::

    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 5s, sys: 1.02 s, total: 2min 6s
    Wall time: 31.7 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7ff8b0e1f970>



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
    Inserting handle into data store.  output_gpz_run: inprogress_output_gpz_run.hdf5, gpz_run


.. parsed-literal::

    Process 0 running estimator on chunk 10,000 - 20,000
    Process 0 estimating GPz PZ PDF for rows 10,000 - 20,000


.. parsed-literal::

    Process 0 running estimator on chunk 20,000 - 20,449
    Process 0 estimating GPz PZ PDF for rows 20,000 - 20,449
    CPU times: user 1.79 s, sys: 44 ms, total: 1.83 s
    Wall time: 581 ms


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

