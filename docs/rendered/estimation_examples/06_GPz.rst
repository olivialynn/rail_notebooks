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
       1	-3.3482561e-01	 3.1783943e-01	-3.2509012e-01	 3.3098969e-01	[-3.5136656e-01]	 4.7309279e-01


.. parsed-literal::

       2	-2.6078111e-01	 3.0579488e-01	-2.3550121e-01	 3.1943843e-01	[-2.7856548e-01]	 2.3497033e-01


.. parsed-literal::

       3	-2.1764440e-01	 2.8681556e-01	-1.7612326e-01	 2.9975078e-01	[-2.3490274e-01]	 2.8288651e-01


.. parsed-literal::

       4	-1.7473259e-01	 2.6814875e-01	-1.2584353e-01	 2.8244199e-01	[-2.0235462e-01]	 3.2305336e-01
       5	-1.3411685e-01	 2.5476908e-01	-1.0452099e-01	 2.6880467e-01	[-1.9418384e-01]	 1.9994712e-01


.. parsed-literal::

       6	-6.3566204e-02	 2.5065811e-01	-3.5284544e-02	 2.6569075e-01	[-9.8161006e-02]	 2.0705533e-01


.. parsed-literal::

       7	-4.5715877e-02	 2.4694979e-01	-2.1191678e-02	 2.6132103e-01	[-7.8273257e-02]	 2.0958185e-01


.. parsed-literal::

       8	-3.4355318e-02	 2.4494702e-01	-1.2591212e-02	 2.5981895e-01	[-7.3043922e-02]	 2.0758247e-01


.. parsed-literal::

       9	-1.9209351e-02	 2.4229114e-01	-1.0958298e-03	 2.5772545e-01	[-6.5995253e-02]	 2.0994568e-01


.. parsed-literal::

      10	-7.6331192e-03	 2.4010056e-01	 8.4364441e-03	 2.5538568e-01	[-5.7234568e-02]	 2.0861697e-01


.. parsed-literal::

      11	-7.6730798e-04	 2.3920721e-01	 1.3315730e-02	 2.5358680e-01	[-4.8586511e-02]	 2.1165800e-01


.. parsed-literal::

      12	 4.1656310e-03	 2.3816092e-01	 1.7918001e-02	 2.5226435e-01	[-4.3031145e-02]	 2.1478295e-01


.. parsed-literal::

      13	 6.4963559e-03	 2.3769854e-01	 2.0166510e-02	 2.5174635e-01	[-4.0503239e-02]	 2.2446251e-01


.. parsed-literal::

      14	 1.4813435e-02	 2.3601485e-01	 2.9274715e-02	 2.5008534e-01	[-3.2129592e-02]	 2.1212006e-01


.. parsed-literal::

      15	 1.3791764e-01	 2.2211204e-01	 1.5917435e-01	 2.3245026e-01	[ 1.2725569e-01]	 3.3070993e-01


.. parsed-literal::

      16	 1.9036195e-01	 2.1817485e-01	 2.1267032e-01	 2.3033444e-01	[ 1.8300751e-01]	 2.2063518e-01


.. parsed-literal::

      17	 2.8018955e-01	 2.1610155e-01	 3.1118223e-01	 2.2914752e-01	[ 2.6587973e-01]	 2.2182345e-01


.. parsed-literal::

      18	 3.1246150e-01	 2.1361661e-01	 3.4407234e-01	 2.2806694e-01	[ 2.9283981e-01]	 2.1940660e-01
      19	 3.6415391e-01	 2.0939826e-01	 3.9705433e-01	 2.2520212e-01	[ 3.4515818e-01]	 1.8838000e-01


.. parsed-literal::

      20	 4.1072992e-01	 2.0751835e-01	 4.4366979e-01	 2.2210321e-01	[ 4.0163946e-01]	 2.1315026e-01
      21	 4.6946527e-01	 2.0523266e-01	 5.0259218e-01	 2.1969250e-01	[ 4.6231236e-01]	 1.9061661e-01


.. parsed-literal::

      22	 5.8167619e-01	 2.0495318e-01	 6.1911066e-01	 2.1754154e-01	[ 5.7600874e-01]	 2.1882701e-01


.. parsed-literal::

      23	 5.9119241e-01	 2.0886684e-01	 6.3222749e-01	 2.2196496e-01	  5.6403740e-01 	 2.1504092e-01


.. parsed-literal::

      24	 6.3496373e-01	 2.0431698e-01	 6.7261837e-01	 2.1631723e-01	[ 6.2307581e-01]	 2.1536446e-01


.. parsed-literal::

      25	 6.6343071e-01	 2.0072673e-01	 7.0126410e-01	 2.1295911e-01	[ 6.5063124e-01]	 2.2296309e-01


.. parsed-literal::

      26	 6.8692609e-01	 1.9823475e-01	 7.2435417e-01	 2.0947774e-01	[ 6.8016195e-01]	 2.1240664e-01


.. parsed-literal::

      27	 7.2232839e-01	 1.9543563e-01	 7.5967913e-01	 2.0400536e-01	[ 7.1617577e-01]	 2.0921731e-01


.. parsed-literal::

      28	 7.5078110e-01	 1.9479665e-01	 7.8898842e-01	 2.0542902e-01	[ 7.2994593e-01]	 2.1403241e-01


.. parsed-literal::

      29	 7.8771943e-01	 1.9567836e-01	 8.2688367e-01	 2.0682161e-01	[ 7.6124066e-01]	 2.1284342e-01


.. parsed-literal::

      30	 8.1204463e-01	 1.9655784e-01	 8.5147237e-01	 2.0870939e-01	[ 7.8538788e-01]	 2.1929264e-01


.. parsed-literal::

      31	 8.2865171e-01	 1.9473009e-01	 8.6837741e-01	 2.0721853e-01	[ 7.9801508e-01]	 2.2309327e-01


.. parsed-literal::

      32	 8.4756781e-01	 1.9378275e-01	 8.8801051e-01	 2.0653800e-01	[ 8.1162734e-01]	 2.1176267e-01


.. parsed-literal::

      33	 8.6692333e-01	 1.9300715e-01	 9.0808357e-01	 2.0651223e-01	[ 8.2932217e-01]	 2.0390463e-01
      34	 8.7862125e-01	 1.9507925e-01	 9.2056569e-01	 2.0968887e-01	[ 8.4584943e-01]	 1.8231344e-01


.. parsed-literal::

      35	 8.9978942e-01	 1.9496135e-01	 9.4211386e-01	 2.1002231e-01	[ 8.7275863e-01]	 2.1577787e-01
      36	 9.1145593e-01	 1.9128771e-01	 9.5347330e-01	 2.0581256e-01	[ 8.8139370e-01]	 1.6479611e-01


.. parsed-literal::

      37	 9.2715250e-01	 1.8759737e-01	 9.6902969e-01	 2.0096551e-01	[ 8.9591381e-01]	 2.0726109e-01
      38	 9.4969313e-01	 1.8525080e-01	 9.9246518e-01	 1.9736298e-01	[ 9.1961836e-01]	 1.7683387e-01


.. parsed-literal::

      39	 9.6682192e-01	 1.8354598e-01	 1.0104537e+00	 1.9484738e-01	[ 9.4123149e-01]	 2.0512819e-01
      40	 9.7969170e-01	 1.8286317e-01	 1.0235657e+00	 1.9436119e-01	[ 9.5828423e-01]	 1.7654395e-01


.. parsed-literal::

      41	 9.9749753e-01	 1.8061676e-01	 1.0421976e+00	 1.9202497e-01	[ 9.7466648e-01]	 1.7882347e-01


.. parsed-literal::

      42	 1.0155105e+00	 1.7737852e-01	 1.0609962e+00	 1.8894286e-01	[ 9.8309764e-01]	 2.1599722e-01


.. parsed-literal::

      43	 1.0192058e+00	 1.7485947e-01	 1.0662357e+00	 1.8608733e-01	  9.7917561e-01 	 2.1600890e-01


.. parsed-literal::

      44	 1.0368913e+00	 1.7323467e-01	 1.0832403e+00	 1.8464654e-01	[ 9.9447465e-01]	 2.0849037e-01
      45	 1.0423915e+00	 1.7254647e-01	 1.0884733e+00	 1.8375639e-01	[ 1.0012444e+00]	 1.9941401e-01


.. parsed-literal::

      46	 1.0533783e+00	 1.7075137e-01	 1.0996036e+00	 1.8158814e-01	[ 1.0104615e+00]	 2.2051930e-01


.. parsed-literal::

      47	 1.0655604e+00	 1.6884842e-01	 1.1120194e+00	 1.8014102e-01	[ 1.0158701e+00]	 2.1421289e-01


.. parsed-literal::

      48	 1.0768192e+00	 1.6725592e-01	 1.1236525e+00	 1.7902510e-01	[ 1.0199700e+00]	 2.1786928e-01


.. parsed-literal::

      49	 1.0881258e+00	 1.6570247e-01	 1.1351960e+00	 1.7860981e-01	[ 1.0245953e+00]	 2.1077919e-01
      50	 1.1006928e+00	 1.6466369e-01	 1.1478603e+00	 1.7906403e-01	[ 1.0296806e+00]	 1.9862509e-01


.. parsed-literal::

      51	 1.1126018e+00	 1.6330321e-01	 1.1596038e+00	 1.7807718e-01	[ 1.0451287e+00]	 2.0335150e-01


.. parsed-literal::

      52	 1.1215519e+00	 1.6009414e-01	 1.1686437e+00	 1.7488946e-01	[ 1.0512071e+00]	 2.0816183e-01


.. parsed-literal::

      53	 1.1307372e+00	 1.5851747e-01	 1.1776343e+00	 1.7260876e-01	[ 1.0612679e+00]	 2.1929216e-01


.. parsed-literal::

      54	 1.1412793e+00	 1.5566296e-01	 1.1883966e+00	 1.6918355e-01	[ 1.0674124e+00]	 2.1581697e-01


.. parsed-literal::

      55	 1.1513869e+00	 1.5384348e-01	 1.1988738e+00	 1.6662277e-01	[ 1.0712115e+00]	 2.0992589e-01


.. parsed-literal::

      56	 1.1620736e+00	 1.5144034e-01	 1.2098097e+00	 1.6426343e-01	[ 1.0799604e+00]	 2.1743393e-01


.. parsed-literal::

      57	 1.1716505e+00	 1.4991602e-01	 1.2195360e+00	 1.6361536e-01	[ 1.0865299e+00]	 2.1596670e-01


.. parsed-literal::

      58	 1.1802663e+00	 1.4868946e-01	 1.2286663e+00	 1.6273668e-01	[ 1.0879958e+00]	 2.1033144e-01


.. parsed-literal::

      59	 1.1883652e+00	 1.4736577e-01	 1.2369245e+00	 1.6137547e-01	[ 1.0897393e+00]	 2.1841049e-01


.. parsed-literal::

      60	 1.1989304e+00	 1.4613352e-01	 1.2475853e+00	 1.5949645e-01	[ 1.0966997e+00]	 2.1394873e-01


.. parsed-literal::

      61	 1.2121350e+00	 1.4381074e-01	 1.2613458e+00	 1.5602600e-01	[ 1.1065885e+00]	 2.0931792e-01
      62	 1.2141775e+00	 1.4375708e-01	 1.2638886e+00	 1.5525310e-01	  1.0845793e+00 	 1.7154193e-01


.. parsed-literal::

      63	 1.2270996e+00	 1.4273675e-01	 1.2762589e+00	 1.5431202e-01	[ 1.1109050e+00]	 2.1155500e-01


.. parsed-literal::

      64	 1.2323248e+00	 1.4192195e-01	 1.2815624e+00	 1.5331078e-01	[ 1.1145638e+00]	 2.1424961e-01
      65	 1.2409376e+00	 1.4119992e-01	 1.2903266e+00	 1.5242351e-01	  1.1143759e+00 	 1.9876385e-01


.. parsed-literal::

      66	 1.2448144e+00	 1.4055299e-01	 1.2946692e+00	 1.5152472e-01	  1.1064705e+00 	 2.1182060e-01


.. parsed-literal::

      67	 1.2523155e+00	 1.4071088e-01	 1.3019857e+00	 1.5167800e-01	  1.1070603e+00 	 2.1199036e-01


.. parsed-literal::

      68	 1.2559353e+00	 1.4069232e-01	 1.3057071e+00	 1.5166746e-01	  1.1036175e+00 	 2.1056247e-01


.. parsed-literal::

      69	 1.2619242e+00	 1.4053756e-01	 1.3118914e+00	 1.5162943e-01	  1.0967179e+00 	 2.0864058e-01


.. parsed-literal::

      70	 1.2710435e+00	 1.3992019e-01	 1.3214261e+00	 1.5091716e-01	  1.0850234e+00 	 2.1380258e-01


.. parsed-literal::

      71	 1.2762421e+00	 1.3922952e-01	 1.3268182e+00	 1.5029212e-01	  1.0704172e+00 	 3.3921170e-01
      72	 1.2821899e+00	 1.3850743e-01	 1.3328900e+00	 1.4919028e-01	  1.0692398e+00 	 1.8083477e-01


.. parsed-literal::

      73	 1.2888168e+00	 1.3766099e-01	 1.3394944e+00	 1.4768308e-01	  1.0714120e+00 	 2.1400166e-01


.. parsed-literal::

      74	 1.2968738e+00	 1.3689622e-01	 1.3476902e+00	 1.4592988e-01	  1.0691887e+00 	 2.1067858e-01
      75	 1.3014003e+00	 1.3587258e-01	 1.3525624e+00	 1.4302899e-01	  1.0775561e+00 	 1.9948530e-01


.. parsed-literal::

      76	 1.3081966e+00	 1.3595652e-01	 1.3592474e+00	 1.4354162e-01	  1.0812428e+00 	 2.1319580e-01


.. parsed-literal::

      77	 1.3121052e+00	 1.3595053e-01	 1.3632172e+00	 1.4362271e-01	  1.0850582e+00 	 2.2204041e-01
      78	 1.3174666e+00	 1.3570440e-01	 1.3687033e+00	 1.4322419e-01	  1.0940666e+00 	 1.9168854e-01


.. parsed-literal::

      79	 1.3245109e+00	 1.3539630e-01	 1.3759802e+00	 1.4229360e-01	  1.1034272e+00 	 2.1386337e-01
      80	 1.3322566e+00	 1.3482024e-01	 1.3838931e+00	 1.4135685e-01	[ 1.1194370e+00]	 1.9966698e-01


.. parsed-literal::

      81	 1.3378597e+00	 1.3428578e-01	 1.3893574e+00	 1.4061257e-01	[ 1.1240927e+00]	 2.0345831e-01


.. parsed-literal::

      82	 1.3447554e+00	 1.3377833e-01	 1.3962995e+00	 1.3972714e-01	  1.1224499e+00 	 2.1371961e-01


.. parsed-literal::

      83	 1.3497964e+00	 1.3364358e-01	 1.4014562e+00	 1.3979140e-01	[ 1.1272280e+00]	 2.1371913e-01
      84	 1.3550200e+00	 1.3358519e-01	 1.4066445e+00	 1.3979916e-01	[ 1.1296550e+00]	 2.0122099e-01


.. parsed-literal::

      85	 1.3612970e+00	 1.3358710e-01	 1.4129937e+00	 1.3996805e-01	[ 1.1346362e+00]	 1.8856382e-01


.. parsed-literal::

      86	 1.3668292e+00	 1.3336943e-01	 1.4186550e+00	 1.3984305e-01	[ 1.1377227e+00]	 2.2083712e-01
      87	 1.3732200e+00	 1.3298971e-01	 1.4253852e+00	 1.4018306e-01	  1.1329864e+00 	 1.8374157e-01


.. parsed-literal::

      88	 1.3790694e+00	 1.3258811e-01	 1.4312341e+00	 1.3987580e-01	[ 1.1422887e+00]	 2.0337319e-01


.. parsed-literal::

      89	 1.3824468e+00	 1.3214172e-01	 1.4345330e+00	 1.3932253e-01	[ 1.1442308e+00]	 2.1350169e-01


.. parsed-literal::

      90	 1.3885506e+00	 1.3139571e-01	 1.4407144e+00	 1.3842089e-01	[ 1.1493025e+00]	 2.1390843e-01
      91	 1.3937975e+00	 1.3072362e-01	 1.4460604e+00	 1.3746203e-01	[ 1.1652240e+00]	 1.8727994e-01


.. parsed-literal::

      92	 1.3980812e+00	 1.3054717e-01	 1.4503025e+00	 1.3723621e-01	[ 1.1724626e+00]	 1.9882369e-01


.. parsed-literal::

      93	 1.4020965e+00	 1.3048147e-01	 1.4543037e+00	 1.3712221e-01	[ 1.1755136e+00]	 2.1153641e-01


.. parsed-literal::

      94	 1.4061095e+00	 1.3038082e-01	 1.4584366e+00	 1.3687622e-01	  1.1745878e+00 	 2.0259094e-01
      95	 1.4095085e+00	 1.3021673e-01	 1.4621557e+00	 1.3675433e-01	  1.1701776e+00 	 1.8637562e-01


.. parsed-literal::

      96	 1.4147398e+00	 1.2996076e-01	 1.4673888e+00	 1.3632525e-01	  1.1729164e+00 	 2.0937419e-01


.. parsed-literal::

      97	 1.4179241e+00	 1.2970456e-01	 1.4705109e+00	 1.3604247e-01	  1.1754409e+00 	 2.1665192e-01


.. parsed-literal::

      98	 1.4221026e+00	 1.2926999e-01	 1.4747448e+00	 1.3537976e-01	[ 1.1791775e+00]	 2.0255804e-01


.. parsed-literal::

      99	 1.4259589e+00	 1.2895175e-01	 1.4787607e+00	 1.3491314e-01	[ 1.1882474e+00]	 2.0958185e-01
     100	 1.4304942e+00	 1.2880948e-01	 1.4832184e+00	 1.3452685e-01	[ 1.1942851e+00]	 1.8104434e-01


.. parsed-literal::

     101	 1.4337172e+00	 1.2879692e-01	 1.4864470e+00	 1.3430258e-01	[ 1.1990422e+00]	 2.0864010e-01
     102	 1.4375397e+00	 1.2878306e-01	 1.4902910e+00	 1.3413905e-01	[ 1.2035498e+00]	 1.7763042e-01


.. parsed-literal::

     103	 1.4423128e+00	 1.2868070e-01	 1.4950927e+00	 1.3380558e-01	[ 1.2179643e+00]	 2.0709968e-01


.. parsed-literal::

     104	 1.4467063e+00	 1.2842945e-01	 1.4994359e+00	 1.3363478e-01	[ 1.2211092e+00]	 2.0751286e-01


.. parsed-literal::

     105	 1.4500934e+00	 1.2827682e-01	 1.5027759e+00	 1.3360469e-01	[ 1.2237221e+00]	 2.1337128e-01


.. parsed-literal::

     106	 1.4554244e+00	 1.2794842e-01	 1.5081480e+00	 1.3360461e-01	[ 1.2273633e+00]	 2.2289276e-01


.. parsed-literal::

     107	 1.4571247e+00	 1.2770486e-01	 1.5100003e+00	 1.3364152e-01	[ 1.2330852e+00]	 2.0428252e-01


.. parsed-literal::

     108	 1.4606457e+00	 1.2765775e-01	 1.5134362e+00	 1.3357714e-01	[ 1.2358379e+00]	 2.0409083e-01


.. parsed-literal::

     109	 1.4639301e+00	 1.2755901e-01	 1.5167351e+00	 1.3342729e-01	[ 1.2397803e+00]	 2.2125268e-01


.. parsed-literal::

     110	 1.4663332e+00	 1.2745832e-01	 1.5191946e+00	 1.3325255e-01	[ 1.2430749e+00]	 2.0939016e-01


.. parsed-literal::

     111	 1.4708851e+00	 1.2731316e-01	 1.5238024e+00	 1.3300903e-01	[ 1.2488248e+00]	 2.1219230e-01


.. parsed-literal::

     112	 1.4729702e+00	 1.2728681e-01	 1.5259886e+00	 1.3264504e-01	  1.2471181e+00 	 3.2024145e-01
     113	 1.4764983e+00	 1.2723484e-01	 1.5295674e+00	 1.3253608e-01	[ 1.2498717e+00]	 2.0012236e-01


.. parsed-literal::

     114	 1.4794236e+00	 1.2719665e-01	 1.5325442e+00	 1.3235862e-01	  1.2483723e+00 	 2.1927476e-01


.. parsed-literal::

     115	 1.4822448e+00	 1.2700275e-01	 1.5355141e+00	 1.3171766e-01	  1.2398249e+00 	 2.0235157e-01


.. parsed-literal::

     116	 1.4843418e+00	 1.2694133e-01	 1.5376817e+00	 1.3149596e-01	  1.2331130e+00 	 2.0260334e-01


.. parsed-literal::

     117	 1.4861843e+00	 1.2682175e-01	 1.5394801e+00	 1.3129284e-01	  1.2338567e+00 	 2.0814466e-01


.. parsed-literal::

     118	 1.4900420e+00	 1.2651814e-01	 1.5433692e+00	 1.3066481e-01	  1.2343023e+00 	 2.0937037e-01
     119	 1.4929658e+00	 1.2626419e-01	 1.5463029e+00	 1.3025708e-01	  1.2349065e+00 	 1.9582701e-01


.. parsed-literal::

     120	 1.4961268e+00	 1.2613290e-01	 1.5494727e+00	 1.2991860e-01	  1.2360193e+00 	 2.1584558e-01


.. parsed-literal::

     121	 1.4987356e+00	 1.2598578e-01	 1.5521079e+00	 1.2966847e-01	  1.2409663e+00 	 2.0354557e-01


.. parsed-literal::

     122	 1.5005615e+00	 1.2602175e-01	 1.5539425e+00	 1.2966680e-01	  1.2381773e+00 	 2.2309899e-01
     123	 1.5025022e+00	 1.2596726e-01	 1.5559674e+00	 1.2955222e-01	  1.2336457e+00 	 1.9935799e-01


.. parsed-literal::

     124	 1.5058829e+00	 1.2567369e-01	 1.5595993e+00	 1.2903595e-01	  1.2218684e+00 	 2.0058513e-01


.. parsed-literal::

     125	 1.5082168e+00	 1.2547701e-01	 1.5620329e+00	 1.2895869e-01	  1.2159781e+00 	 2.1864533e-01


.. parsed-literal::

     126	 1.5100193e+00	 1.2531026e-01	 1.5637443e+00	 1.2895485e-01	  1.2217276e+00 	 2.1484804e-01
     127	 1.5119944e+00	 1.2509338e-01	 1.5656164e+00	 1.2903096e-01	  1.2241107e+00 	 1.8342924e-01


.. parsed-literal::

     128	 1.5135699e+00	 1.2485294e-01	 1.5671807e+00	 1.2923729e-01	  1.2223770e+00 	 2.1249795e-01


.. parsed-literal::

     129	 1.5156094e+00	 1.2472249e-01	 1.5692020e+00	 1.2948797e-01	  1.2171627e+00 	 2.1222711e-01


.. parsed-literal::

     130	 1.5181524e+00	 1.2436019e-01	 1.5718765e+00	 1.2994029e-01	  1.2026962e+00 	 2.0288157e-01
     131	 1.5206961e+00	 1.2423951e-01	 1.5744964e+00	 1.2992252e-01	  1.1927308e+00 	 2.0275116e-01


.. parsed-literal::

     132	 1.5219162e+00	 1.2422799e-01	 1.5756878e+00	 1.2971311e-01	  1.1961218e+00 	 2.1331239e-01


.. parsed-literal::

     133	 1.5244914e+00	 1.2408413e-01	 1.5783171e+00	 1.2961951e-01	  1.1954152e+00 	 2.2051716e-01


.. parsed-literal::

     134	 1.5266983e+00	 1.2389396e-01	 1.5805890e+00	 1.2961504e-01	  1.1914379e+00 	 2.0567751e-01


.. parsed-literal::

     135	 1.5286503e+00	 1.2363655e-01	 1.5826097e+00	 1.2999381e-01	  1.1877042e+00 	 2.1184564e-01


.. parsed-literal::

     136	 1.5304958e+00	 1.2354877e-01	 1.5843910e+00	 1.3004792e-01	  1.1839462e+00 	 2.2140861e-01


.. parsed-literal::

     137	 1.5319966e+00	 1.2342629e-01	 1.5859032e+00	 1.3014571e-01	  1.1774129e+00 	 2.1363425e-01


.. parsed-literal::

     138	 1.5333433e+00	 1.2330370e-01	 1.5873301e+00	 1.3035931e-01	  1.1718175e+00 	 2.1395326e-01


.. parsed-literal::

     139	 1.5350168e+00	 1.2315226e-01	 1.5891883e+00	 1.3070544e-01	  1.1588498e+00 	 2.1056247e-01


.. parsed-literal::

     140	 1.5363693e+00	 1.2311321e-01	 1.5905631e+00	 1.3073190e-01	  1.1559684e+00 	 2.1973300e-01


.. parsed-literal::

     141	 1.5373557e+00	 1.2308380e-01	 1.5915774e+00	 1.3080329e-01	  1.1550297e+00 	 2.1374607e-01


.. parsed-literal::

     142	 1.5389725e+00	 1.2305962e-01	 1.5932321e+00	 1.3102513e-01	  1.1483171e+00 	 2.1573877e-01


.. parsed-literal::

     143	 1.5398974e+00	 1.2301687e-01	 1.5943297e+00	 1.3154865e-01	  1.1386279e+00 	 2.1581483e-01


.. parsed-literal::

     144	 1.5414620e+00	 1.2299322e-01	 1.5957536e+00	 1.3166305e-01	  1.1362150e+00 	 2.1800780e-01


.. parsed-literal::

     145	 1.5421617e+00	 1.2297399e-01	 1.5964228e+00	 1.3176103e-01	  1.1342224e+00 	 2.0823860e-01
     146	 1.5434518e+00	 1.2294274e-01	 1.5977050e+00	 1.3204916e-01	  1.1304623e+00 	 1.7645931e-01


.. parsed-literal::

     147	 1.5449092e+00	 1.2295892e-01	 1.5992695e+00	 1.3245172e-01	  1.1206484e+00 	 2.1444392e-01


.. parsed-literal::

     148	 1.5466134e+00	 1.2294728e-01	 1.6009233e+00	 1.3276673e-01	  1.1181230e+00 	 2.1332955e-01


.. parsed-literal::

     149	 1.5475841e+00	 1.2293856e-01	 1.6019155e+00	 1.3287166e-01	  1.1146050e+00 	 2.2084260e-01
     150	 1.5486272e+00	 1.2293112e-01	 1.6030134e+00	 1.3289341e-01	  1.1082951e+00 	 1.8594933e-01


.. parsed-literal::

    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 7s, sys: 1.17 s, total: 2min 8s
    Wall time: 32.2 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fd6f0605ba0>



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
    CPU times: user 1.84 s, sys: 64 ms, total: 1.9 s
    Wall time: 615 ms


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

