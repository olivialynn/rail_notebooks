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
       1	-3.3367748e-01	 3.1705734e-01	-3.2397518e-01	 3.3510043e-01	[-3.5847470e-01]	 4.7205973e-01


.. parsed-literal::

       2	-2.6141255e-01	 3.0639519e-01	-2.3744307e-01	 3.2259236e-01	[-2.8835034e-01]	 2.3384571e-01


.. parsed-literal::

       3	-2.1370593e-01	 2.8482914e-01	-1.7104730e-01	 3.0044972e-01	[-2.3749676e-01]	 2.8039908e-01


.. parsed-literal::

       4	-1.7546931e-01	 2.6779442e-01	-1.2716375e-01	 2.8428681e-01	[-2.0859555e-01]	 2.8951979e-01


.. parsed-literal::

       5	-1.1625907e-01	 2.5486496e-01	-8.4711857e-02	 2.7224129e-01	[-1.8150169e-01]	 2.1010494e-01


.. parsed-literal::

       6	-6.2304071e-02	 2.5024405e-01	-3.3726964e-02	 2.6665486e-01	[-9.9013125e-02]	 2.1129990e-01


.. parsed-literal::

       7	-4.3266803e-02	 2.4638289e-01	-1.8919081e-02	 2.6230272e-01	[-8.2423532e-02]	 2.1382785e-01


.. parsed-literal::

       8	-3.0244449e-02	 2.4429034e-01	-9.7229153e-03	 2.5981601e-01	[-7.6017918e-02]	 2.1111345e-01


.. parsed-literal::

       9	-1.4763366e-02	 2.4135012e-01	 2.2736877e-03	 2.5667449e-01	[-5.8916717e-02]	 2.1344686e-01
      10	-3.9728716e-03	 2.3944349e-01	 1.1338484e-02	 2.5522966e-01	[-5.7473355e-02]	 1.8302751e-01


.. parsed-literal::

      11	 1.0350041e-03	 2.3843705e-01	 1.5817220e-02	 2.5478839e-01	[-5.5654131e-02]	 2.0923162e-01
      12	 6.9956472e-03	 2.3740884e-01	 2.1010143e-02	 2.5412456e-01	[-5.0530108e-02]	 1.7576385e-01


.. parsed-literal::

      13	 1.2872301e-02	 2.3645065e-01	 2.6570420e-02	 2.5318726e-01	[-4.4128329e-02]	 2.0350385e-01


.. parsed-literal::

      14	 8.4491443e-02	 2.2382017e-01	 1.0407189e-01	 2.3966897e-01	[ 4.9842344e-02]	 3.2432747e-01


.. parsed-literal::

      15	 1.0313676e-01	 2.2195567e-01	 1.2454116e-01	 2.3816743e-01	[ 8.0223324e-02]	 4.4661283e-01


.. parsed-literal::

      16	 1.5501548e-01	 2.1903893e-01	 1.7638274e-01	 2.3643656e-01	[ 1.2684795e-01]	 2.0946741e-01


.. parsed-literal::

      17	 2.5244999e-01	 2.1584834e-01	 2.8066518e-01	 2.2942568e-01	[ 2.2604124e-01]	 2.0675492e-01


.. parsed-literal::

      18	 2.9435755e-01	 2.1389379e-01	 3.2675788e-01	 2.2919473e-01	[ 2.5888504e-01]	 2.1967530e-01
      19	 3.3921394e-01	 2.1253728e-01	 3.7181654e-01	 2.2829955e-01	[ 3.0643128e-01]	 1.9006658e-01


.. parsed-literal::

      20	 3.9815327e-01	 2.0779559e-01	 4.3119229e-01	 2.2504743e-01	[ 3.6640794e-01]	 2.1426153e-01


.. parsed-literal::

      21	 4.6266319e-01	 2.0279669e-01	 4.9652591e-01	 2.1917190e-01	[ 4.3052666e-01]	 2.0920753e-01


.. parsed-literal::

      22	 5.4469136e-01	 2.0238638e-01	 5.8048980e-01	 2.1879964e-01	[ 5.0478852e-01]	 2.1037388e-01


.. parsed-literal::

      23	 6.0905398e-01	 2.0040652e-01	 6.4828903e-01	 2.1499318e-01	[ 5.6690692e-01]	 2.1667552e-01


.. parsed-literal::

      24	 6.4944374e-01	 1.9618369e-01	 6.8902960e-01	 2.1006102e-01	[ 6.1488574e-01]	 2.2113729e-01
      25	 6.7838691e-01	 1.9447456e-01	 7.1706336e-01	 2.0861269e-01	[ 6.4940356e-01]	 2.0223498e-01


.. parsed-literal::

      26	 7.0529529e-01	 1.9221182e-01	 7.4419047e-01	 2.0679213e-01	[ 6.8022960e-01]	 2.1223807e-01


.. parsed-literal::

      27	 7.3603819e-01	 1.9155466e-01	 7.7468633e-01	 2.0685981e-01	[ 7.0882246e-01]	 2.0229602e-01


.. parsed-literal::

      28	 7.5867606e-01	 1.9432283e-01	 7.9720782e-01	 2.1010645e-01	[ 7.3220298e-01]	 2.1332836e-01


.. parsed-literal::

      29	 7.8625472e-01	 1.9359149e-01	 8.2554375e-01	 2.0991075e-01	[ 7.5816098e-01]	 2.1125102e-01


.. parsed-literal::

      30	 8.0101316e-01	 2.0575907e-01	 8.4197459e-01	 2.1988648e-01	  7.4592594e-01 	 2.0816135e-01


.. parsed-literal::

      31	 8.4625119e-01	 2.0121526e-01	 8.8678280e-01	 2.1921197e-01	[ 7.9819283e-01]	 2.1199393e-01
      32	 8.6214576e-01	 1.9966269e-01	 9.0276016e-01	 2.1722267e-01	[ 8.2087494e-01]	 1.9205999e-01


.. parsed-literal::

      33	 8.7529855e-01	 1.9910868e-01	 9.1641016e-01	 2.1687488e-01	[ 8.3283345e-01]	 2.1185350e-01


.. parsed-literal::

      34	 9.1898185e-01	 1.9845983e-01	 9.6124333e-01	 2.1500928e-01	[ 8.6880003e-01]	 2.1474743e-01
      35	 9.2428548e-01	 1.9309523e-01	 9.6909606e-01	 2.0941645e-01	  8.5127104e-01 	 1.9685960e-01


.. parsed-literal::

      36	 9.5236927e-01	 1.9192108e-01	 9.9615755e-01	 2.0787173e-01	[ 8.8340156e-01]	 2.1987867e-01


.. parsed-literal::

      37	 9.6715362e-01	 1.8838721e-01	 1.0112646e+00	 2.0347186e-01	[ 9.0351224e-01]	 2.1612430e-01


.. parsed-literal::

      38	 9.8006458e-01	 1.8516018e-01	 1.0242454e+00	 1.9903564e-01	[ 9.1889369e-01]	 2.2093368e-01


.. parsed-literal::

      39	 1.0023779e+00	 1.7980787e-01	 1.0473056e+00	 1.9331358e-01	[ 9.3701290e-01]	 2.1265674e-01


.. parsed-literal::

      40	 1.0170557e+00	 1.7456606e-01	 1.0632592e+00	 1.8758518e-01	[ 9.6242350e-01]	 2.1466041e-01
      41	 1.0349024e+00	 1.7487095e-01	 1.0807656e+00	 1.8919121e-01	[ 9.7717697e-01]	 1.9886971e-01


.. parsed-literal::

      42	 1.0471972e+00	 1.7561792e-01	 1.0931217e+00	 1.9032857e-01	[ 9.9240401e-01]	 2.0138860e-01


.. parsed-literal::

      43	 1.0624080e+00	 1.7535076e-01	 1.1088243e+00	 1.9047405e-01	[ 1.0052119e+00]	 2.2120500e-01


.. parsed-literal::

      44	 1.0808723e+00	 1.7347152e-01	 1.1277622e+00	 1.8723447e-01	[ 1.0371087e+00]	 2.1763420e-01


.. parsed-literal::

      45	 1.0928514e+00	 1.7106954e-01	 1.1399565e+00	 1.8315112e-01	[ 1.0475369e+00]	 2.0723104e-01
      46	 1.1033157e+00	 1.6872798e-01	 1.1505766e+00	 1.7983974e-01	[ 1.0572976e+00]	 1.9659328e-01


.. parsed-literal::

      47	 1.1194286e+00	 1.6590486e-01	 1.1668649e+00	 1.7550191e-01	[ 1.0681041e+00]	 2.0960188e-01
      48	 1.1333719e+00	 1.6230844e-01	 1.1812286e+00	 1.7090888e-01	[ 1.0737431e+00]	 1.9028521e-01


.. parsed-literal::

      49	 1.1468045e+00	 1.6098573e-01	 1.1943709e+00	 1.6949336e-01	[ 1.0860365e+00]	 2.0377088e-01


.. parsed-literal::

      50	 1.1564669e+00	 1.5945714e-01	 1.2042908e+00	 1.6833668e-01	[ 1.0928562e+00]	 2.1659255e-01


.. parsed-literal::

      51	 1.1668058e+00	 1.5874638e-01	 1.2149326e+00	 1.6831803e-01	[ 1.1019721e+00]	 2.1686196e-01


.. parsed-literal::

      52	 1.1759751e+00	 1.5643683e-01	 1.2243366e+00	 1.6657253e-01	  1.0989057e+00 	 2.0333767e-01
      53	 1.1840497e+00	 1.5561366e-01	 1.2324335e+00	 1.6542824e-01	[ 1.1078287e+00]	 1.7180729e-01


.. parsed-literal::

      54	 1.1983100e+00	 1.5295786e-01	 1.2473934e+00	 1.6261673e-01	  1.1064877e+00 	 2.0510387e-01


.. parsed-literal::

      55	 1.2121346e+00	 1.5128845e-01	 1.2614339e+00	 1.6078285e-01	  1.1054036e+00 	 2.0849872e-01
      56	 1.2235584e+00	 1.4908889e-01	 1.2738032e+00	 1.5982788e-01	  1.0851341e+00 	 2.0805216e-01


.. parsed-literal::

      57	 1.2363678e+00	 1.4832875e-01	 1.2864640e+00	 1.5834115e-01	  1.1006350e+00 	 2.0952582e-01


.. parsed-literal::

      58	 1.2459340e+00	 1.4721884e-01	 1.2960957e+00	 1.5664310e-01	[ 1.1137154e+00]	 2.0373440e-01


.. parsed-literal::

      59	 1.2594981e+00	 1.4543079e-01	 1.3097618e+00	 1.5442607e-01	[ 1.1284770e+00]	 2.1342826e-01
      60	 1.2680930e+00	 1.4335310e-01	 1.3193076e+00	 1.5152596e-01	  1.1254254e+00 	 1.7600703e-01


.. parsed-literal::

      61	 1.2792675e+00	 1.4361118e-01	 1.3300437e+00	 1.5277205e-01	[ 1.1399105e+00]	 2.0148921e-01
      62	 1.2855389e+00	 1.4335155e-01	 1.3361972e+00	 1.5308608e-01	  1.1393032e+00 	 2.0082092e-01


.. parsed-literal::

      63	 1.2921850e+00	 1.4320365e-01	 1.3429244e+00	 1.5365310e-01	[ 1.1417996e+00]	 2.1617842e-01
      64	 1.2978020e+00	 1.4262881e-01	 1.3492682e+00	 1.5247804e-01	  1.1256312e+00 	 2.0136070e-01


.. parsed-literal::

      65	 1.3064199e+00	 1.4315441e-01	 1.3575436e+00	 1.5371002e-01	  1.1408001e+00 	 2.0614147e-01


.. parsed-literal::

      66	 1.3114300e+00	 1.4274435e-01	 1.3625534e+00	 1.5344760e-01	[ 1.1490127e+00]	 2.1219921e-01


.. parsed-literal::

      67	 1.3210222e+00	 1.4179977e-01	 1.3724183e+00	 1.5278313e-01	[ 1.1510352e+00]	 2.0997977e-01


.. parsed-literal::

      68	 1.3301465e+00	 1.3966898e-01	 1.3817374e+00	 1.5101234e-01	[ 1.1520256e+00]	 2.1157217e-01


.. parsed-literal::

      69	 1.3391682e+00	 1.3881211e-01	 1.3909287e+00	 1.5088019e-01	  1.1444855e+00 	 2.1229696e-01


.. parsed-literal::

      70	 1.3438221e+00	 1.3866997e-01	 1.3953726e+00	 1.5044856e-01	  1.1510256e+00 	 2.0441651e-01
      71	 1.3497488e+00	 1.3806543e-01	 1.4015065e+00	 1.4995172e-01	[ 1.1520498e+00]	 1.8257546e-01


.. parsed-literal::

      72	 1.3550276e+00	 1.3726789e-01	 1.4069858e+00	 1.4956600e-01	[ 1.1556850e+00]	 2.0669746e-01


.. parsed-literal::

      73	 1.3610306e+00	 1.3676882e-01	 1.4130126e+00	 1.4907026e-01	[ 1.1627559e+00]	 2.0637751e-01
      74	 1.3672830e+00	 1.3631790e-01	 1.4193310e+00	 1.4792875e-01	[ 1.1757306e+00]	 2.0520401e-01


.. parsed-literal::

      75	 1.3731364e+00	 1.3600349e-01	 1.4253847e+00	 1.4786606e-01	[ 1.1810662e+00]	 2.1181870e-01
      76	 1.3783929e+00	 1.3514299e-01	 1.4307782e+00	 1.4506314e-01	[ 1.1876460e+00]	 1.7669535e-01


.. parsed-literal::

      77	 1.3821440e+00	 1.3486116e-01	 1.4345241e+00	 1.4471660e-01	[ 1.1906558e+00]	 1.9339037e-01


.. parsed-literal::

      78	 1.3874798e+00	 1.3422937e-01	 1.4399811e+00	 1.4402410e-01	  1.1867511e+00 	 2.0559907e-01
      79	 1.3939932e+00	 1.3365393e-01	 1.4466723e+00	 1.4329457e-01	  1.1833254e+00 	 2.0335531e-01


.. parsed-literal::

      80	 1.4005607e+00	 1.3320725e-01	 1.4532253e+00	 1.4244679e-01	  1.1821759e+00 	 2.1115017e-01


.. parsed-literal::

      81	 1.4056326e+00	 1.3291150e-01	 1.4582428e+00	 1.4142193e-01	  1.1844591e+00 	 2.1783948e-01


.. parsed-literal::

      82	 1.4094221e+00	 1.3282196e-01	 1.4621279e+00	 1.4065315e-01	  1.1859782e+00 	 2.0774150e-01
      83	 1.4133570e+00	 1.3275492e-01	 1.4660260e+00	 1.4065919e-01	  1.1900159e+00 	 2.0465684e-01


.. parsed-literal::

      84	 1.4164355e+00	 1.3269320e-01	 1.4691444e+00	 1.4045468e-01	[ 1.1910483e+00]	 2.1607065e-01
      85	 1.4219529e+00	 1.3273202e-01	 1.4746994e+00	 1.4035634e-01	[ 1.2015015e+00]	 1.8495059e-01


.. parsed-literal::

      86	 1.4256373e+00	 1.3294823e-01	 1.4787071e+00	 1.3984517e-01	  1.1942144e+00 	 2.0966339e-01


.. parsed-literal::

      87	 1.4315175e+00	 1.3301103e-01	 1.4844329e+00	 1.4019929e-01	[ 1.2101251e+00]	 2.1721244e-01


.. parsed-literal::

      88	 1.4347490e+00	 1.3313749e-01	 1.4876416e+00	 1.4047295e-01	[ 1.2188635e+00]	 2.1979880e-01
      89	 1.4370462e+00	 1.3335608e-01	 1.4900149e+00	 1.4064996e-01	[ 1.2192581e+00]	 1.9529009e-01


.. parsed-literal::

      90	 1.4403035e+00	 1.3348771e-01	 1.4933974e+00	 1.4102114e-01	[ 1.2225238e+00]	 2.0630717e-01
      91	 1.4439759e+00	 1.3355244e-01	 1.4971517e+00	 1.4142574e-01	[ 1.2233399e+00]	 1.9532633e-01


.. parsed-literal::

      92	 1.4488842e+00	 1.3331936e-01	 1.5021946e+00	 1.4172974e-01	[ 1.2233922e+00]	 2.1998668e-01


.. parsed-literal::

      93	 1.4516510e+00	 1.3330879e-01	 1.5050418e+00	 1.4367285e-01	  1.2214606e+00 	 2.1129417e-01


.. parsed-literal::

      94	 1.4546158e+00	 1.3299083e-01	 1.5078451e+00	 1.4264771e-01	[ 1.2266860e+00]	 2.0697308e-01


.. parsed-literal::

      95	 1.4569591e+00	 1.3281163e-01	 1.5101678e+00	 1.4225414e-01	[ 1.2274772e+00]	 2.1614647e-01


.. parsed-literal::

      96	 1.4603648e+00	 1.3272486e-01	 1.5136500e+00	 1.4222316e-01	  1.2253793e+00 	 2.0189667e-01


.. parsed-literal::

      97	 1.4633659e+00	 1.3270155e-01	 1.5168407e+00	 1.4233680e-01	  1.2172482e+00 	 2.1285081e-01


.. parsed-literal::

      98	 1.4664472e+00	 1.3275974e-01	 1.5199101e+00	 1.4275119e-01	  1.2164440e+00 	 2.1109605e-01


.. parsed-literal::

      99	 1.4690083e+00	 1.3280842e-01	 1.5225111e+00	 1.4316612e-01	  1.2132036e+00 	 2.0222664e-01


.. parsed-literal::

     100	 1.4712311e+00	 1.3275513e-01	 1.5247785e+00	 1.4339887e-01	  1.2068482e+00 	 2.1325922e-01


.. parsed-literal::

     101	 1.4750680e+00	 1.3268672e-01	 1.5288220e+00	 1.4410966e-01	  1.1943955e+00 	 2.1327519e-01


.. parsed-literal::

     102	 1.4774808e+00	 1.3284582e-01	 1.5313423e+00	 1.4435344e-01	  1.1729401e+00 	 2.1336722e-01


.. parsed-literal::

     103	 1.4796041e+00	 1.3264208e-01	 1.5333187e+00	 1.4391637e-01	  1.1853748e+00 	 2.1085548e-01
     104	 1.4811975e+00	 1.3254437e-01	 1.5349026e+00	 1.4385049e-01	  1.1880009e+00 	 1.9845939e-01


.. parsed-literal::

     105	 1.4838311e+00	 1.3240198e-01	 1.5375508e+00	 1.4381362e-01	  1.1875715e+00 	 2.0745158e-01


.. parsed-literal::

     106	 1.4874815e+00	 1.3221380e-01	 1.5412525e+00	 1.4396305e-01	  1.1795310e+00 	 2.1330261e-01


.. parsed-literal::

     107	 1.4893800e+00	 1.3203992e-01	 1.5432035e+00	 1.4408249e-01	  1.1785927e+00 	 2.9771161e-01


.. parsed-literal::

     108	 1.4919310e+00	 1.3199482e-01	 1.5458387e+00	 1.4440983e-01	  1.1631847e+00 	 2.1286583e-01


.. parsed-literal::

     109	 1.4933618e+00	 1.3191649e-01	 1.5473053e+00	 1.4435333e-01	  1.1613450e+00 	 2.1752119e-01


.. parsed-literal::

     110	 1.4948958e+00	 1.3187267e-01	 1.5488422e+00	 1.4430786e-01	  1.1609746e+00 	 2.1047950e-01


.. parsed-literal::

     111	 1.4969161e+00	 1.3176643e-01	 1.5509063e+00	 1.4422580e-01	  1.1632565e+00 	 2.1427965e-01


.. parsed-literal::

     112	 1.4988953e+00	 1.3167479e-01	 1.5529367e+00	 1.4386700e-01	  1.1674487e+00 	 2.1354389e-01


.. parsed-literal::

     113	 1.5008672e+00	 1.3151273e-01	 1.5550170e+00	 1.4417414e-01	  1.1693431e+00 	 2.1135926e-01


.. parsed-literal::

     114	 1.5026836e+00	 1.3150711e-01	 1.5567614e+00	 1.4411741e-01	  1.1715197e+00 	 2.1520901e-01


.. parsed-literal::

     115	 1.5042764e+00	 1.3140847e-01	 1.5583055e+00	 1.4403657e-01	  1.1702217e+00 	 2.0984864e-01


.. parsed-literal::

     116	 1.5058050e+00	 1.3135094e-01	 1.5598293e+00	 1.4424528e-01	  1.1662557e+00 	 2.1397662e-01


.. parsed-literal::

     117	 1.5078009e+00	 1.3107438e-01	 1.5618812e+00	 1.4423705e-01	  1.1596773e+00 	 2.1938848e-01


.. parsed-literal::

     118	 1.5095764e+00	 1.3090671e-01	 1.5637404e+00	 1.4426782e-01	  1.1567880e+00 	 2.2146964e-01
     119	 1.5115336e+00	 1.3068352e-01	 1.5658333e+00	 1.4429802e-01	  1.1545062e+00 	 1.8987775e-01


.. parsed-literal::

     120	 1.5131917e+00	 1.3053598e-01	 1.5675616e+00	 1.4403027e-01	  1.1584581e+00 	 2.1565628e-01


.. parsed-literal::

     121	 1.5149644e+00	 1.3051487e-01	 1.5693227e+00	 1.4421283e-01	  1.1592414e+00 	 2.1050000e-01


.. parsed-literal::

     122	 1.5169326e+00	 1.3041205e-01	 1.5712686e+00	 1.4410972e-01	  1.1590547e+00 	 2.1697974e-01


.. parsed-literal::

     123	 1.5182823e+00	 1.3043807e-01	 1.5725935e+00	 1.4425713e-01	  1.1552890e+00 	 2.2274923e-01


.. parsed-literal::

     124	 1.5203612e+00	 1.3028449e-01	 1.5746746e+00	 1.4427083e-01	  1.1469856e+00 	 2.1017408e-01


.. parsed-literal::

     125	 1.5221680e+00	 1.3019961e-01	 1.5764980e+00	 1.4383263e-01	  1.1319586e+00 	 2.1158791e-01
     126	 1.5234881e+00	 1.3013056e-01	 1.5778477e+00	 1.4374264e-01	  1.1261635e+00 	 1.9468498e-01


.. parsed-literal::

     127	 1.5249404e+00	 1.3002858e-01	 1.5794169e+00	 1.4374140e-01	  1.1199748e+00 	 1.8853164e-01


.. parsed-literal::

     128	 1.5263917e+00	 1.2996857e-01	 1.5809486e+00	 1.4337565e-01	  1.1125580e+00 	 2.1337128e-01


.. parsed-literal::

     129	 1.5273171e+00	 1.2993606e-01	 1.5818603e+00	 1.4313690e-01	  1.1157719e+00 	 2.2031975e-01


.. parsed-literal::

     130	 1.5289842e+00	 1.2987844e-01	 1.5835397e+00	 1.4279842e-01	  1.1185383e+00 	 2.1048832e-01


.. parsed-literal::

     131	 1.5299750e+00	 1.2963537e-01	 1.5845909e+00	 1.4185323e-01	  1.1182183e+00 	 2.2352338e-01


.. parsed-literal::

     132	 1.5314254e+00	 1.2964988e-01	 1.5859782e+00	 1.4214110e-01	  1.1177456e+00 	 2.1351147e-01


.. parsed-literal::

     133	 1.5327943e+00	 1.2960902e-01	 1.5873413e+00	 1.4232945e-01	  1.1122048e+00 	 2.1462202e-01
     134	 1.5337595e+00	 1.2957211e-01	 1.5883030e+00	 1.4233735e-01	  1.1085620e+00 	 1.9186640e-01


.. parsed-literal::

     135	 1.5361561e+00	 1.2958162e-01	 1.5907323e+00	 1.4243211e-01	  1.1026447e+00 	 2.3812127e-01


.. parsed-literal::

     136	 1.5374020e+00	 1.2945620e-01	 1.5920045e+00	 1.4234723e-01	  1.0918736e+00 	 3.3001137e-01


.. parsed-literal::

     137	 1.5384959e+00	 1.2947123e-01	 1.5931381e+00	 1.4215577e-01	  1.0881042e+00 	 2.1653795e-01


.. parsed-literal::

     138	 1.5395735e+00	 1.2945496e-01	 1.5942689e+00	 1.4210401e-01	  1.0842322e+00 	 2.1184921e-01


.. parsed-literal::

     139	 1.5407205e+00	 1.2943774e-01	 1.5954962e+00	 1.4212490e-01	  1.0736991e+00 	 2.1298003e-01


.. parsed-literal::

     140	 1.5418780e+00	 1.2943430e-01	 1.5967202e+00	 1.4222837e-01	  1.0622781e+00 	 2.1125150e-01


.. parsed-literal::

     141	 1.5429424e+00	 1.2939976e-01	 1.5977612e+00	 1.4234748e-01	  1.0571563e+00 	 2.1388888e-01
     142	 1.5442456e+00	 1.2936813e-01	 1.5990227e+00	 1.4232086e-01	  1.0481419e+00 	 1.9797587e-01


.. parsed-literal::

     143	 1.5456183e+00	 1.2936282e-01	 1.6003949e+00	 1.4221925e-01	  1.0402190e+00 	 2.2198772e-01


.. parsed-literal::

     144	 1.5475750e+00	 1.2932558e-01	 1.6023998e+00	 1.4156636e-01	  1.0269895e+00 	 2.1093035e-01


.. parsed-literal::

     145	 1.5487968e+00	 1.2948927e-01	 1.6036749e+00	 1.4175990e-01	  1.0211436e+00 	 2.1291876e-01


.. parsed-literal::

     146	 1.5498437e+00	 1.2944849e-01	 1.6047065e+00	 1.4165451e-01	  1.0295988e+00 	 2.0731783e-01


.. parsed-literal::

     147	 1.5509744e+00	 1.2945175e-01	 1.6058800e+00	 1.4161339e-01	  1.0347901e+00 	 2.2148705e-01


.. parsed-literal::

     148	 1.5520533e+00	 1.2949402e-01	 1.6070489e+00	 1.4160175e-01	  1.0237927e+00 	 2.0943928e-01


.. parsed-literal::

     149	 1.5533712e+00	 1.2964676e-01	 1.6084664e+00	 1.4177338e-01	  1.0109831e+00 	 2.0667672e-01


.. parsed-literal::

     150	 1.5545136e+00	 1.2969807e-01	 1.6096115e+00	 1.4179907e-01	  9.9258791e-01 	 2.1055198e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 8s, sys: 1.16 s, total: 2min 9s
    Wall time: 32.5 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f2800e1bac0>



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
    Inserting handle into data store.  output_gpz_run: inprogress_output_gpz_run.hdf5, gpz_run


.. parsed-literal::

    Process 0 running estimator on chunk 10,000 - 20,000
    Process 0 estimating GPz PZ PDF for rows 10,000 - 20,000


.. parsed-literal::

    Process 0 running estimator on chunk 20,000 - 20,449
    Process 0 estimating GPz PZ PDF for rows 20,000 - 20,449
    CPU times: user 2.24 s, sys: 41 ms, total: 2.28 s
    Wall time: 738 ms


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

