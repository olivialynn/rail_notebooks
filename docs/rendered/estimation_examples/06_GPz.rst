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
       1	-3.2597616e-01	 3.1539648e-01	-3.1634246e-01	 3.4037399e-01	[-3.6664492e-01]	 4.6393228e-01


.. parsed-literal::

       2	-2.5984821e-01	 3.0654966e-01	-2.3768533e-01	 3.2876161e-01	[-3.0847030e-01]	 2.3905683e-01


.. parsed-literal::

       3	-2.1468257e-01	 2.8503783e-01	-1.7317220e-01	 2.9662359e-01	[-2.2663828e-01]	 2.8817677e-01


.. parsed-literal::

       4	-1.7419270e-01	 2.6304349e-01	-1.3364703e-01	 2.7095661e-01	[-1.8270802e-01]	 2.0527291e-01
       5	-8.4710932e-02	 2.5334139e-01	-5.3803828e-02	 2.6253131e-01	[-9.8688777e-02]	 2.0305920e-01


.. parsed-literal::

       6	-5.8155375e-02	 2.4931679e-01	-3.0501908e-02	 2.5971000e-01	[-7.0610882e-02]	 2.0876765e-01


.. parsed-literal::

       7	-4.0109675e-02	 2.4613469e-01	-1.7492269e-02	 2.5769328e-01	[-6.3656037e-02]	 2.0621109e-01
       8	-2.7856800e-02	 2.4406057e-01	-8.7036280e-03	 2.5672605e-01	[-6.1097895e-02]	 1.9626951e-01


.. parsed-literal::

       9	-1.6689765e-02	 2.4199325e-01	 2.3146678e-04	 2.5543042e-01	[-5.5568166e-02]	 2.0298195e-01
      10	-7.7441910e-03	 2.3998415e-01	 7.4840514e-03	 2.5278188e-01	[-5.1139490e-02]	 1.7454720e-01


.. parsed-literal::

      11	-2.7623111e-03	 2.3925353e-01	 1.1297128e-02	 2.5120181e-01	[-3.5182585e-02]	 2.1119809e-01


.. parsed-literal::

      12	 8.7498012e-04	 2.3880713e-01	 1.4900538e-02	 2.5049991e-01	[-3.3686789e-02]	 2.0246577e-01


.. parsed-literal::

      13	 3.6411147e-03	 2.3822935e-01	 1.7748753e-02	 2.4964296e-01	[-3.1286372e-02]	 2.0886874e-01


.. parsed-literal::

      14	 8.1424070e-03	 2.3717644e-01	 2.3067066e-02	 2.4788691e-01	[-2.3888445e-02]	 2.0551467e-01


.. parsed-literal::

      15	 2.0229443e-01	 2.2105665e-01	 2.2910038e-01	 2.3062073e-01	[ 1.9974369e-01]	 3.2241797e-01
      16	 2.2158283e-01	 2.2257597e-01	 2.4939193e-01	 2.3066764e-01	[ 2.3458004e-01]	 1.9945478e-01


.. parsed-literal::

      17	 2.9965211e-01	 2.1551847e-01	 3.2902986e-01	 2.2251448e-01	[ 3.1734790e-01]	 2.0576620e-01


.. parsed-literal::

      18	 3.9388284e-01	 2.1133326e-01	 4.2693105e-01	 2.1802019e-01	[ 4.1887525e-01]	 2.1525073e-01
      19	 4.4798118e-01	 2.0787795e-01	 4.8255205e-01	 2.1307961e-01	[ 4.8788975e-01]	 2.0282722e-01


.. parsed-literal::

      20	 5.0597756e-01	 2.0463456e-01	 5.4206088e-01	 2.1054216e-01	[ 5.5517862e-01]	 2.0502567e-01


.. parsed-literal::

      21	 5.6623501e-01	 2.0029322e-01	 6.0442195e-01	 2.0707964e-01	[ 6.1966683e-01]	 2.0495605e-01


.. parsed-literal::

      22	 6.1433448e-01	 1.9601272e-01	 6.5345847e-01	 2.0323806e-01	[ 6.5476825e-01]	 2.0731044e-01


.. parsed-literal::

      23	 6.6007616e-01	 1.9214253e-01	 6.9986723e-01	 1.9753229e-01	[ 7.0440842e-01]	 2.0463061e-01
      24	 7.0792323e-01	 1.9022614e-01	 7.4746049e-01	 1.9418683e-01	[ 7.3726420e-01]	 1.9582295e-01


.. parsed-literal::

      25	 7.5797793e-01	 1.9368672e-01	 7.9640336e-01	 1.9742738e-01	[ 7.7393200e-01]	 3.1120396e-01


.. parsed-literal::

      26	 7.8840134e-01	 1.9564792e-01	 8.2757645e-01	 1.9879487e-01	[ 8.0732225e-01]	 2.1632171e-01


.. parsed-literal::

      27	 8.1912334e-01	 1.9252260e-01	 8.6042263e-01	 1.9892144e-01	[ 8.1986686e-01]	 2.1145535e-01


.. parsed-literal::

      28	 8.4626850e-01	 1.8928927e-01	 8.8696979e-01	 1.9762579e-01	[ 8.3945230e-01]	 2.0871782e-01
      29	 8.7265589e-01	 1.8736208e-01	 9.1395468e-01	 1.9744941e-01	[ 8.6014731e-01]	 1.9820189e-01


.. parsed-literal::

      30	 8.9364744e-01	 1.8777233e-01	 9.3630398e-01	 1.9951869e-01	[ 8.7890648e-01]	 1.9046044e-01


.. parsed-literal::

      31	 9.1367578e-01	 1.8698561e-01	 9.5665241e-01	 1.9779190e-01	[ 8.9987038e-01]	 2.1769667e-01
      32	 9.3485537e-01	 1.8537004e-01	 9.7880919e-01	 1.9554610e-01	[ 9.2608818e-01]	 1.9715810e-01


.. parsed-literal::

      33	 9.5960146e-01	 1.8361536e-01	 1.0043455e+00	 1.9416271e-01	[ 9.5489340e-01]	 2.0947456e-01


.. parsed-literal::

      34	 9.7983102e-01	 1.7984901e-01	 1.0250031e+00	 1.8960492e-01	[ 9.7934249e-01]	 2.1827555e-01


.. parsed-literal::

      35	 9.9582524e-01	 1.7665800e-01	 1.0409665e+00	 1.8572686e-01	[ 9.9836785e-01]	 2.0919967e-01


.. parsed-literal::

      36	 1.0092718e+00	 1.7239683e-01	 1.0552218e+00	 1.7907380e-01	[ 1.0247483e+00]	 2.0442557e-01
      37	 1.0254763e+00	 1.7098922e-01	 1.0714138e+00	 1.7796781e-01	[ 1.0380492e+00]	 1.9504356e-01


.. parsed-literal::

      38	 1.0350676e+00	 1.6936628e-01	 1.0813458e+00	 1.7579464e-01	[ 1.0491834e+00]	 1.8510771e-01


.. parsed-literal::

      39	 1.0505327e+00	 1.6522201e-01	 1.0975269e+00	 1.7075584e-01	[ 1.0614750e+00]	 2.1078157e-01


.. parsed-literal::

      40	 1.0686575e+00	 1.5981689e-01	 1.1164261e+00	 1.6396197e-01	[ 1.0806186e+00]	 2.0447683e-01


.. parsed-literal::

      41	 1.0832390e+00	 1.5601728e-01	 1.1310600e+00	 1.5937979e-01	[ 1.0911239e+00]	 2.1785331e-01


.. parsed-literal::

      42	 1.0982528e+00	 1.5490868e-01	 1.1460242e+00	 1.5756603e-01	[ 1.1074908e+00]	 2.1063757e-01


.. parsed-literal::

      43	 1.1162470e+00	 1.5322254e-01	 1.1642077e+00	 1.5488423e-01	[ 1.1288288e+00]	 2.1100783e-01


.. parsed-literal::

      44	 1.1280959e+00	 1.5181892e-01	 1.1759275e+00	 1.5426188e-01	[ 1.1341130e+00]	 2.1731734e-01


.. parsed-literal::

      45	 1.1403338e+00	 1.5092671e-01	 1.1879240e+00	 1.5471344e-01	[ 1.1428070e+00]	 2.1124935e-01


.. parsed-literal::

      46	 1.1499911e+00	 1.4979854e-01	 1.1978441e+00	 1.5423591e-01	[ 1.1498273e+00]	 2.1857738e-01


.. parsed-literal::

      47	 1.1654571e+00	 1.4854703e-01	 1.2141317e+00	 1.5462224e-01	[ 1.1575786e+00]	 2.0801973e-01
      48	 1.1788330e+00	 1.4726463e-01	 1.2280238e+00	 1.5479593e-01	[ 1.1644094e+00]	 1.8949652e-01


.. parsed-literal::

      49	 1.1908903e+00	 1.4615939e-01	 1.2405781e+00	 1.5525278e-01	[ 1.1673821e+00]	 2.1545863e-01


.. parsed-literal::

      50	 1.2014713e+00	 1.4505310e-01	 1.2511717e+00	 1.5449833e-01	[ 1.1751677e+00]	 2.0945048e-01
      51	 1.2145222e+00	 1.4401680e-01	 1.2645984e+00	 1.5400719e-01	[ 1.1853385e+00]	 1.8894267e-01


.. parsed-literal::

      52	 1.2266533e+00	 1.4325932e-01	 1.2766413e+00	 1.5512691e-01	[ 1.1877187e+00]	 2.1815419e-01
      53	 1.2349513e+00	 1.4322675e-01	 1.2850072e+00	 1.5618702e-01	[ 1.1928982e+00]	 1.8822789e-01


.. parsed-literal::

      54	 1.2485343e+00	 1.4298816e-01	 1.2990530e+00	 1.5820543e-01	[ 1.1991265e+00]	 1.9733047e-01


.. parsed-literal::

      55	 1.2572469e+00	 1.4263888e-01	 1.3082649e+00	 1.5749806e-01	[ 1.2109060e+00]	 2.1567225e-01


.. parsed-literal::

      56	 1.2669794e+00	 1.4200645e-01	 1.3177633e+00	 1.5654189e-01	[ 1.2211252e+00]	 2.2309923e-01


.. parsed-literal::

      57	 1.2764087e+00	 1.4162180e-01	 1.3273301e+00	 1.5583100e-01	[ 1.2306682e+00]	 2.1851087e-01


.. parsed-literal::

      58	 1.2843511e+00	 1.4115512e-01	 1.3354450e+00	 1.5448820e-01	[ 1.2424970e+00]	 2.1698999e-01


.. parsed-literal::

      59	 1.2948061e+00	 1.4197641e-01	 1.3459182e+00	 1.5687756e-01	[ 1.2550090e+00]	 2.1982026e-01


.. parsed-literal::

      60	 1.3062376e+00	 1.4127028e-01	 1.3575925e+00	 1.5632058e-01	[ 1.2669025e+00]	 2.0854402e-01


.. parsed-literal::

      61	 1.3124901e+00	 1.4077660e-01	 1.3639136e+00	 1.5620262e-01	[ 1.2703344e+00]	 2.1467352e-01


.. parsed-literal::

      62	 1.3258391e+00	 1.4005715e-01	 1.3774324e+00	 1.5711358e-01	[ 1.2730562e+00]	 2.0993900e-01


.. parsed-literal::

      63	 1.3301162e+00	 1.3946835e-01	 1.3823262e+00	 1.5676055e-01	  1.2717469e+00 	 2.1989894e-01


.. parsed-literal::

      64	 1.3408137e+00	 1.3964658e-01	 1.3927038e+00	 1.5748964e-01	[ 1.2788890e+00]	 2.0995355e-01


.. parsed-literal::

      65	 1.3455406e+00	 1.3924713e-01	 1.3973033e+00	 1.5650820e-01	[ 1.2877765e+00]	 2.1198869e-01


.. parsed-literal::

      66	 1.3523940e+00	 1.3924592e-01	 1.4044856e+00	 1.5666486e-01	[ 1.2903023e+00]	 2.0964384e-01
      67	 1.3612547e+00	 1.3933551e-01	 1.4135080e+00	 1.5644241e-01	[ 1.2981515e+00]	 1.7794538e-01


.. parsed-literal::

      68	 1.3664456e+00	 1.3922469e-01	 1.4192430e+00	 1.5638905e-01	  1.2935256e+00 	 2.0829058e-01


.. parsed-literal::

      69	 1.3736777e+00	 1.3906294e-01	 1.4262791e+00	 1.5633922e-01	[ 1.3016189e+00]	 2.1428919e-01


.. parsed-literal::

      70	 1.3782439e+00	 1.3870016e-01	 1.4308920e+00	 1.5582918e-01	[ 1.3066390e+00]	 2.1276331e-01


.. parsed-literal::

      71	 1.3842536e+00	 1.3824083e-01	 1.4369041e+00	 1.5547960e-01	[ 1.3114580e+00]	 2.2252679e-01


.. parsed-literal::

      72	 1.3904821e+00	 1.3732357e-01	 1.4436601e+00	 1.5368844e-01	[ 1.3213349e+00]	 2.1985435e-01


.. parsed-literal::

      73	 1.3965698e+00	 1.3708358e-01	 1.4497281e+00	 1.5435656e-01	  1.3185892e+00 	 2.0662713e-01


.. parsed-literal::

      74	 1.3995468e+00	 1.3694971e-01	 1.4525973e+00	 1.5401941e-01	[ 1.3227468e+00]	 2.1313643e-01


.. parsed-literal::

      75	 1.4046581e+00	 1.3670879e-01	 1.4578659e+00	 1.5378648e-01	[ 1.3245682e+00]	 2.1844482e-01


.. parsed-literal::

      76	 1.4092880e+00	 1.3626140e-01	 1.4624885e+00	 1.5378638e-01	[ 1.3292992e+00]	 2.1134543e-01
      77	 1.4147508e+00	 1.3626868e-01	 1.4679251e+00	 1.5426230e-01	[ 1.3293324e+00]	 2.0121264e-01


.. parsed-literal::

      78	 1.4193758e+00	 1.3636827e-01	 1.4725587e+00	 1.5486109e-01	  1.3293187e+00 	 2.0684290e-01


.. parsed-literal::

      79	 1.4243518e+00	 1.3631765e-01	 1.4777520e+00	 1.5544438e-01	  1.3241301e+00 	 2.1448207e-01


.. parsed-literal::

      80	 1.4280695e+00	 1.3654451e-01	 1.4816133e+00	 1.5595674e-01	  1.3197041e+00 	 2.1139956e-01


.. parsed-literal::

      81	 1.4310246e+00	 1.3634207e-01	 1.4845076e+00	 1.5575978e-01	  1.3207995e+00 	 2.2143006e-01


.. parsed-literal::

      82	 1.4350160e+00	 1.3619907e-01	 1.4885045e+00	 1.5579722e-01	  1.3174868e+00 	 2.1326089e-01


.. parsed-literal::

      83	 1.4384564e+00	 1.3593857e-01	 1.4920170e+00	 1.5548768e-01	  1.3155275e+00 	 2.1160650e-01


.. parsed-literal::

      84	 1.4428396e+00	 1.3598865e-01	 1.4964160e+00	 1.5572269e-01	  1.3121201e+00 	 2.1075964e-01


.. parsed-literal::

      85	 1.4467010e+00	 1.3625171e-01	 1.5004298e+00	 1.5673352e-01	  1.3048307e+00 	 2.1873713e-01
      86	 1.4491821e+00	 1.3603893e-01	 1.5030139e+00	 1.5626438e-01	  1.3063587e+00 	 1.9748521e-01


.. parsed-literal::

      87	 1.4514345e+00	 1.3592499e-01	 1.5052475e+00	 1.5634947e-01	  1.3098080e+00 	 2.0831823e-01


.. parsed-literal::

      88	 1.4543445e+00	 1.3570450e-01	 1.5081766e+00	 1.5624611e-01	  1.3134486e+00 	 2.1595955e-01


.. parsed-literal::

      89	 1.4573372e+00	 1.3567570e-01	 1.5111684e+00	 1.5642602e-01	  1.3161459e+00 	 2.1422124e-01


.. parsed-literal::

      90	 1.4604019e+00	 1.3560108e-01	 1.5141922e+00	 1.5630213e-01	  1.3189620e+00 	 2.1824360e-01


.. parsed-literal::

      91	 1.4629926e+00	 1.3556823e-01	 1.5167816e+00	 1.5600761e-01	  1.3187256e+00 	 2.1374369e-01


.. parsed-literal::

      92	 1.4665589e+00	 1.3566966e-01	 1.5203531e+00	 1.5586936e-01	  1.3177727e+00 	 2.2382593e-01


.. parsed-literal::

      93	 1.4690718e+00	 1.3561086e-01	 1.5230160e+00	 1.5546469e-01	  1.3153815e+00 	 2.1367502e-01
      94	 1.4718951e+00	 1.3544913e-01	 1.5258152e+00	 1.5544127e-01	  1.3159298e+00 	 1.9877601e-01


.. parsed-literal::

      95	 1.4742542e+00	 1.3525910e-01	 1.5281829e+00	 1.5552076e-01	  1.3182282e+00 	 2.2150302e-01


.. parsed-literal::

      96	 1.4765649e+00	 1.3492441e-01	 1.5305171e+00	 1.5529924e-01	  1.3207076e+00 	 2.1570373e-01


.. parsed-literal::

      97	 1.4793565e+00	 1.3463247e-01	 1.5333256e+00	 1.5534543e-01	  1.3279216e+00 	 2.0879507e-01


.. parsed-literal::

      98	 1.4817270e+00	 1.3445618e-01	 1.5356923e+00	 1.5522400e-01	[ 1.3315199e+00]	 2.0675421e-01


.. parsed-literal::

      99	 1.4840192e+00	 1.3435806e-01	 1.5380297e+00	 1.5515348e-01	[ 1.3343882e+00]	 2.2035789e-01


.. parsed-literal::

     100	 1.4864194e+00	 1.3426326e-01	 1.5405586e+00	 1.5535642e-01	  1.3330600e+00 	 2.1932983e-01


.. parsed-literal::

     101	 1.4892389e+00	 1.3433856e-01	 1.5435660e+00	 1.5564030e-01	  1.3308203e+00 	 2.1188116e-01
     102	 1.4915515e+00	 1.3425269e-01	 1.5459122e+00	 1.5584470e-01	  1.3261131e+00 	 1.9817567e-01


.. parsed-literal::

     103	 1.4932773e+00	 1.3410868e-01	 1.5476329e+00	 1.5572822e-01	  1.3242022e+00 	 2.0906925e-01


.. parsed-literal::

     104	 1.4952749e+00	 1.3395289e-01	 1.5496341e+00	 1.5563576e-01	  1.3223591e+00 	 2.0184326e-01
     105	 1.4976856e+00	 1.3355614e-01	 1.5520974e+00	 1.5521381e-01	  1.3213821e+00 	 2.0079613e-01


.. parsed-literal::

     106	 1.4996957e+00	 1.3343982e-01	 1.5541236e+00	 1.5523503e-01	  1.3228194e+00 	 2.0257425e-01


.. parsed-literal::

     107	 1.5020053e+00	 1.3337724e-01	 1.5564638e+00	 1.5533969e-01	  1.3246362e+00 	 2.1765184e-01


.. parsed-literal::

     108	 1.5038935e+00	 1.3347443e-01	 1.5584171e+00	 1.5581719e-01	  1.3220432e+00 	 2.0902109e-01


.. parsed-literal::

     109	 1.5058979e+00	 1.3354642e-01	 1.5604091e+00	 1.5598132e-01	  1.3228291e+00 	 2.1338582e-01


.. parsed-literal::

     110	 1.5081651e+00	 1.3359863e-01	 1.5626962e+00	 1.5604856e-01	  1.3215730e+00 	 2.1162295e-01


.. parsed-literal::

     111	 1.5095728e+00	 1.3364013e-01	 1.5641404e+00	 1.5617793e-01	  1.3227107e+00 	 2.2120881e-01


.. parsed-literal::

     112	 1.5112194e+00	 1.3357228e-01	 1.5657803e+00	 1.5612091e-01	  1.3242349e+00 	 2.2046423e-01


.. parsed-literal::

     113	 1.5137555e+00	 1.3341154e-01	 1.5683509e+00	 1.5604124e-01	  1.3254221e+00 	 2.0727491e-01


.. parsed-literal::

     114	 1.5149502e+00	 1.3352924e-01	 1.5695953e+00	 1.5626637e-01	  1.3266375e+00 	 2.1020699e-01


.. parsed-literal::

     115	 1.5165161e+00	 1.3357521e-01	 1.5711919e+00	 1.5647855e-01	  1.3264363e+00 	 2.1663189e-01


.. parsed-literal::

     116	 1.5185097e+00	 1.3374491e-01	 1.5732697e+00	 1.5690097e-01	  1.3240135e+00 	 2.1334958e-01


.. parsed-literal::

     117	 1.5199525e+00	 1.3375149e-01	 1.5747753e+00	 1.5719158e-01	  1.3217151e+00 	 2.1211529e-01
     118	 1.5220216e+00	 1.3363286e-01	 1.5769071e+00	 1.5730767e-01	  1.3194013e+00 	 1.7678547e-01


.. parsed-literal::

     119	 1.5236322e+00	 1.3343663e-01	 1.5785725e+00	 1.5757088e-01	  1.3158021e+00 	 2.0435739e-01


.. parsed-literal::

     120	 1.5253896e+00	 1.3315627e-01	 1.5802977e+00	 1.5728012e-01	  1.3177367e+00 	 2.1998024e-01


.. parsed-literal::

     121	 1.5265462e+00	 1.3295751e-01	 1.5814154e+00	 1.5712386e-01	  1.3188940e+00 	 2.2019410e-01
     122	 1.5282763e+00	 1.3277906e-01	 1.5831476e+00	 1.5712830e-01	  1.3171945e+00 	 2.0014215e-01


.. parsed-literal::

     123	 1.5310382e+00	 1.3235451e-01	 1.5859610e+00	 1.5687353e-01	  1.3125625e+00 	 2.1617579e-01


.. parsed-literal::

     124	 1.5324259e+00	 1.3230652e-01	 1.5873887e+00	 1.5703989e-01	  1.3060944e+00 	 3.0497622e-01
     125	 1.5339934e+00	 1.3212059e-01	 1.5889615e+00	 1.5675305e-01	  1.3042511e+00 	 1.8590069e-01


.. parsed-literal::

     126	 1.5354596e+00	 1.3198309e-01	 1.5904534e+00	 1.5652746e-01	  1.3008558e+00 	 2.0582294e-01


.. parsed-literal::

     127	 1.5367049e+00	 1.3186972e-01	 1.5917529e+00	 1.5618540e-01	  1.2976770e+00 	 2.1747851e-01


.. parsed-literal::

     128	 1.5382902e+00	 1.3178594e-01	 1.5933432e+00	 1.5610948e-01	  1.2950299e+00 	 2.1275306e-01
     129	 1.5400042e+00	 1.3174535e-01	 1.5950584e+00	 1.5609371e-01	  1.2924751e+00 	 1.9648957e-01


.. parsed-literal::

     130	 1.5414846e+00	 1.3164762e-01	 1.5965438e+00	 1.5599712e-01	  1.2895416e+00 	 2.0472717e-01


.. parsed-literal::

     131	 1.5424563e+00	 1.3155499e-01	 1.5975864e+00	 1.5563542e-01	  1.2872123e+00 	 2.1599984e-01


.. parsed-literal::

     132	 1.5440463e+00	 1.3143671e-01	 1.5991290e+00	 1.5562428e-01	  1.2873961e+00 	 2.1110940e-01
     133	 1.5448490e+00	 1.3132638e-01	 1.5999310e+00	 1.5549167e-01	  1.2879792e+00 	 1.8768120e-01


.. parsed-literal::

     134	 1.5460014e+00	 1.3114105e-01	 1.6011040e+00	 1.5528858e-01	  1.2866785e+00 	 2.0819759e-01


.. parsed-literal::

     135	 1.5473649e+00	 1.3078347e-01	 1.6025112e+00	 1.5483845e-01	  1.2843279e+00 	 2.1405101e-01


.. parsed-literal::

     136	 1.5488154e+00	 1.3064613e-01	 1.6039429e+00	 1.5480670e-01	  1.2811838e+00 	 2.1315336e-01
     137	 1.5495529e+00	 1.3065835e-01	 1.6046441e+00	 1.5485908e-01	  1.2812131e+00 	 1.7664504e-01


.. parsed-literal::

     138	 1.5503923e+00	 1.3065978e-01	 1.6054493e+00	 1.5489083e-01	  1.2800109e+00 	 2.0264506e-01
     139	 1.5516396e+00	 1.3071076e-01	 1.6066634e+00	 1.5499931e-01	  1.2781031e+00 	 1.9682527e-01


.. parsed-literal::

     140	 1.5526256e+00	 1.3076652e-01	 1.6076626e+00	 1.5526144e-01	  1.2718236e+00 	 2.1672916e-01


.. parsed-literal::

     141	 1.5540818e+00	 1.3075956e-01	 1.6091088e+00	 1.5513882e-01	  1.2725799e+00 	 2.3919177e-01
     142	 1.5547752e+00	 1.3074598e-01	 1.6098231e+00	 1.5510940e-01	  1.2733094e+00 	 1.7868638e-01


.. parsed-literal::

     143	 1.5559574e+00	 1.3069263e-01	 1.6110706e+00	 1.5499360e-01	  1.2729252e+00 	 2.1501398e-01


.. parsed-literal::

     144	 1.5568485e+00	 1.3058167e-01	 1.6121229e+00	 1.5504298e-01	  1.2656607e+00 	 2.1295333e-01


.. parsed-literal::

     145	 1.5585467e+00	 1.3053425e-01	 1.6137972e+00	 1.5476664e-01	  1.2676375e+00 	 2.1605158e-01


.. parsed-literal::

     146	 1.5593306e+00	 1.3048140e-01	 1.6145745e+00	 1.5462372e-01	  1.2664509e+00 	 2.1168089e-01


.. parsed-literal::

     147	 1.5600777e+00	 1.3042105e-01	 1.6153214e+00	 1.5448496e-01	  1.2643781e+00 	 2.1075583e-01


.. parsed-literal::

     148	 1.5611506e+00	 1.3036976e-01	 1.6163945e+00	 1.5438974e-01	  1.2608864e+00 	 2.0979190e-01


.. parsed-literal::

     149	 1.5622242e+00	 1.3029524e-01	 1.6174915e+00	 1.5415057e-01	  1.2590343e+00 	 2.0603752e-01


.. parsed-literal::

     150	 1.5629610e+00	 1.3030313e-01	 1.6182200e+00	 1.5418110e-01	  1.2596128e+00 	 2.1753144e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 7s, sys: 1.07 s, total: 2min 8s
    Wall time: 32.2 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f8c60f2ab60>



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


.. parsed-literal::

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
    CPU times: user 2.18 s, sys: 47 ms, total: 2.23 s
    Wall time: 700 ms


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

