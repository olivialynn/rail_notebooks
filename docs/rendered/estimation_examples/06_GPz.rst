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
       1	-3.3795945e-01	 3.1907341e-01	-3.2823377e-01	 3.2676703e-01	[-3.4217158e-01]	 4.5517874e-01


.. parsed-literal::

       2	-2.6947039e-01	 3.0914305e-01	-2.4638891e-01	 3.1613156e-01	[-2.6733022e-01]	 2.2558308e-01


.. parsed-literal::

       3	-2.2610393e-01	 2.8922646e-01	-1.8576160e-01	 2.9396015e-01	[-2.0615738e-01]	 2.7274227e-01
       4	-1.8969754e-01	 2.6510895e-01	-1.4707632e-01	 2.7012252e-01	[-1.7386024e-01]	 2.0088673e-01


.. parsed-literal::

       5	-1.0454495e-01	 2.5641891e-01	-7.0488710e-02	 2.6387505e-01	[-1.0839085e-01]	 2.0347619e-01


.. parsed-literal::

       6	-6.6566868e-02	 2.5039916e-01	-3.5792309e-02	 2.5842938e-01	[-6.6049216e-02]	 2.0972872e-01
       7	-4.9569670e-02	 2.4787516e-01	-2.4637360e-02	 2.5650019e-01	[-5.9098401e-02]	 2.0202827e-01


.. parsed-literal::

       8	-3.3687800e-02	 2.4513553e-01	-1.3182321e-02	 2.5400711e-01	[-4.9425307e-02]	 1.7977786e-01
       9	-2.0464455e-02	 2.4273305e-01	-2.9673019e-03	 2.5211153e-01	[-4.2512198e-02]	 1.9898057e-01


.. parsed-literal::

      10	-1.1429058e-02	 2.4121057e-01	 3.4835992e-03	 2.4856623e-01	[-2.5266499e-02]	 1.9332433e-01
      11	-4.2795718e-03	 2.3992075e-01	 9.6926272e-03	 2.4732631e-01	[-2.2266376e-02]	 2.0027947e-01


.. parsed-literal::

      12	-1.9377256e-03	 2.3957049e-01	 1.1786611e-02	 2.4707902e-01	[-1.9089291e-02]	 1.9760895e-01
      13	 1.1065105e-03	 2.3903716e-01	 1.4574562e-02	 2.4644945e-01	[-1.5754590e-02]	 1.9690943e-01


.. parsed-literal::

      14	 7.6085643e-03	 2.3769967e-01	 2.1925934e-02	 2.4526073e-01	[-7.9832055e-03]	 2.0671511e-01
      15	 8.0651876e-02	 2.2605741e-01	 9.7830228e-02	 2.3337881e-01	[ 7.7019931e-02]	 1.9831467e-01


.. parsed-literal::

      16	 1.0255396e-01	 2.2447774e-01	 1.2203172e-01	 2.3268996e-01	[ 1.0479418e-01]	 3.2779288e-01


.. parsed-literal::

      17	 1.5978315e-01	 2.2100183e-01	 1.8089453e-01	 2.2939965e-01	[ 1.6771978e-01]	 2.0587039e-01


.. parsed-literal::

      18	 2.6111677e-01	 2.2005872e-01	 2.8935060e-01	 2.2464071e-01	[ 2.7694887e-01]	 2.0681739e-01


.. parsed-literal::

      19	 2.9979295e-01	 2.1822181e-01	 3.2977522e-01	 2.2444041e-01	[ 3.0917179e-01]	 2.0152092e-01


.. parsed-literal::

      20	 3.4094130e-01	 2.1594679e-01	 3.7199291e-01	 2.2535974e-01	[ 3.4676791e-01]	 2.0797014e-01
      21	 3.8976976e-01	 2.1510495e-01	 4.1989500e-01	 2.2445400e-01	[ 3.9386924e-01]	 1.9665813e-01


.. parsed-literal::

      22	 4.6891139e-01	 2.1056415e-01	 5.0037072e-01	 2.1833287e-01	[ 4.7814766e-01]	 2.1473551e-01
      23	 5.7921544e-01	 2.0637643e-01	 6.1442666e-01	 2.1310487e-01	[ 5.9124225e-01]	 1.7921472e-01


.. parsed-literal::

      24	 6.0520183e-01	 2.0549797e-01	 6.4433546e-01	 2.1519868e-01	[ 6.1671415e-01]	 1.8228936e-01


.. parsed-literal::

      25	 6.5900677e-01	 2.0025609e-01	 6.9663823e-01	 2.0906479e-01	[ 6.7232450e-01]	 2.0331740e-01
      26	 6.8444638e-01	 1.9789570e-01	 7.2111304e-01	 2.0675219e-01	[ 6.9617109e-01]	 1.9161367e-01


.. parsed-literal::

      27	 7.2372388e-01	 1.9903881e-01	 7.5946875e-01	 2.1169118e-01	[ 7.2831806e-01]	 2.1387577e-01
      28	 7.4517126e-01	 2.0398677e-01	 7.8105106e-01	 2.1854817e-01	[ 7.5078222e-01]	 2.0054269e-01


.. parsed-literal::

      29	 7.7764112e-01	 1.9527298e-01	 8.1477991e-01	 2.0875557e-01	[ 7.8909964e-01]	 2.0511627e-01
      30	 8.0392281e-01	 1.9370985e-01	 8.4222833e-01	 2.0669092e-01	[ 8.2027273e-01]	 1.7871833e-01


.. parsed-literal::

      31	 8.2808855e-01	 1.9248949e-01	 8.6783779e-01	 2.0671688e-01	[ 8.4745236e-01]	 2.0036578e-01


.. parsed-literal::

      32	 8.5128465e-01	 1.9574342e-01	 8.9114925e-01	 2.1173772e-01	[ 8.7002739e-01]	 2.1110773e-01


.. parsed-literal::

      33	 8.6601270e-01	 1.9317509e-01	 9.0576221e-01	 2.0828506e-01	[ 8.8901892e-01]	 2.0467329e-01


.. parsed-literal::

      34	 8.8262210e-01	 1.9268782e-01	 9.2240025e-01	 2.0883575e-01	[ 9.1214038e-01]	 2.0475841e-01
      35	 8.9481561e-01	 1.9195552e-01	 9.3526513e-01	 2.0869244e-01	[ 9.2995295e-01]	 1.9861269e-01


.. parsed-literal::

      36	 9.1398787e-01	 1.9189868e-01	 9.5518031e-01	 2.1008159e-01	[ 9.5399669e-01]	 2.1182847e-01
      37	 9.2701920e-01	 1.9236431e-01	 9.6987960e-01	 2.1079749e-01	[ 9.6178654e-01]	 1.9071007e-01


.. parsed-literal::

      38	 9.3944947e-01	 1.9172132e-01	 9.8229601e-01	 2.1057926e-01	[ 9.7886033e-01]	 1.8849301e-01


.. parsed-literal::

      39	 9.4728104e-01	 1.8852073e-01	 9.8962036e-01	 2.0698667e-01	[ 9.8662739e-01]	 2.0859337e-01


.. parsed-literal::

      40	 9.5771216e-01	 1.8529764e-01	 9.9995811e-01	 2.0366304e-01	[ 9.9383524e-01]	 2.0657635e-01
      41	 9.6965965e-01	 1.8242940e-01	 1.0124973e+00	 2.0068172e-01	[ 1.0019358e+00]	 1.7202878e-01


.. parsed-literal::

      42	 9.8475737e-01	 1.8104918e-01	 1.0285890e+00	 1.9977594e-01	[ 1.0096778e+00]	 2.0202231e-01


.. parsed-literal::

      43	 9.9672865e-01	 1.7870313e-01	 1.0414245e+00	 1.9737035e-01	[ 1.0174038e+00]	 2.1198940e-01


.. parsed-literal::

      44	 1.0066015e+00	 1.7804813e-01	 1.0514937e+00	 1.9683791e-01	[ 1.0260505e+00]	 2.0959139e-01


.. parsed-literal::

      45	 1.0189034e+00	 1.7743681e-01	 1.0642804e+00	 1.9633136e-01	[ 1.0409885e+00]	 2.1017027e-01
      46	 1.0238914e+00	 1.7845683e-01	 1.0699034e+00	 1.9780801e-01	[ 1.0476434e+00]	 2.0423794e-01


.. parsed-literal::

      47	 1.0329739e+00	 1.7665404e-01	 1.0784769e+00	 1.9559859e-01	[ 1.0568558e+00]	 2.0338464e-01
      48	 1.0393323e+00	 1.7525090e-01	 1.0849843e+00	 1.9388719e-01	[ 1.0616955e+00]	 1.9806409e-01


.. parsed-literal::

      49	 1.0459701e+00	 1.7466389e-01	 1.0919334e+00	 1.9329385e-01	[ 1.0664729e+00]	 1.9764543e-01
      50	 1.0572610e+00	 1.7405474e-01	 1.1037175e+00	 1.9270773e-01	[ 1.0765605e+00]	 2.0876002e-01


.. parsed-literal::

      51	 1.0649852e+00	 1.7432859e-01	 1.1120010e+00	 1.9357229e-01	[ 1.0801116e+00]	 1.9268751e-01


.. parsed-literal::

      52	 1.0736945e+00	 1.7304488e-01	 1.1206262e+00	 1.9192288e-01	[ 1.0887632e+00]	 2.2007227e-01


.. parsed-literal::

      53	 1.0817357e+00	 1.7164640e-01	 1.1285065e+00	 1.9059175e-01	[ 1.0958206e+00]	 2.0795584e-01
      54	 1.0944245e+00	 1.6891670e-01	 1.1410664e+00	 1.8842533e-01	[ 1.1070931e+00]	 1.6655183e-01


.. parsed-literal::

      55	 1.1014554e+00	 1.6662617e-01	 1.1484775e+00	 1.8642975e-01	  1.1062523e+00 	 1.7860866e-01


.. parsed-literal::

      56	 1.1117989e+00	 1.6517645e-01	 1.1586864e+00	 1.8496544e-01	[ 1.1188208e+00]	 2.0289993e-01
      57	 1.1185818e+00	 1.6433153e-01	 1.1655509e+00	 1.8428397e-01	[ 1.1263820e+00]	 1.8758512e-01


.. parsed-literal::

      58	 1.1276961e+00	 1.6306515e-01	 1.1750120e+00	 1.8265053e-01	[ 1.1363616e+00]	 2.1535444e-01
      59	 1.1390787e+00	 1.6071020e-01	 1.1867140e+00	 1.8020268e-01	[ 1.1467353e+00]	 1.7149282e-01


.. parsed-literal::

      60	 1.1480162e+00	 1.6006182e-01	 1.1960542e+00	 1.7813461e-01	[ 1.1564658e+00]	 2.0598817e-01


.. parsed-literal::

      61	 1.1568205e+00	 1.5899222e-01	 1.2047547e+00	 1.7741702e-01	[ 1.1656810e+00]	 2.0785451e-01
      62	 1.1685141e+00	 1.5757860e-01	 1.2169472e+00	 1.7678358e-01	[ 1.1758757e+00]	 1.7926621e-01


.. parsed-literal::

      63	 1.1736064e+00	 1.5651664e-01	 1.2222148e+00	 1.7611653e-01	[ 1.1801302e+00]	 2.0322442e-01


.. parsed-literal::

      64	 1.1801470e+00	 1.5625726e-01	 1.2285838e+00	 1.7569008e-01	[ 1.1870243e+00]	 2.1236181e-01
      65	 1.1869508e+00	 1.5558190e-01	 1.2355432e+00	 1.7481576e-01	[ 1.1916920e+00]	 1.8205762e-01


.. parsed-literal::

      66	 1.1928607e+00	 1.5489964e-01	 1.2417627e+00	 1.7427034e-01	[ 1.1951276e+00]	 1.8804359e-01


.. parsed-literal::

      67	 1.2046936e+00	 1.5328493e-01	 1.2543234e+00	 1.7420507e-01	[ 1.1977026e+00]	 2.0903397e-01


.. parsed-literal::

      68	 1.2087474e+00	 1.5355387e-01	 1.2592579e+00	 1.7515023e-01	[ 1.2032671e+00]	 2.1674466e-01


.. parsed-literal::

      69	 1.2182183e+00	 1.5202491e-01	 1.2681927e+00	 1.7419302e-01	[ 1.2103085e+00]	 2.0597935e-01


.. parsed-literal::

      70	 1.2241700e+00	 1.5140518e-01	 1.2740983e+00	 1.7392744e-01	[ 1.2148380e+00]	 2.1189976e-01
      71	 1.2320049e+00	 1.5078128e-01	 1.2821264e+00	 1.7349299e-01	[ 1.2237663e+00]	 1.9976091e-01


.. parsed-literal::

      72	 1.2376503e+00	 1.5079450e-01	 1.2882378e+00	 1.7348204e-01	[ 1.2268745e+00]	 2.1612072e-01


.. parsed-literal::

      73	 1.2461695e+00	 1.5048899e-01	 1.2967528e+00	 1.7303253e-01	[ 1.2391436e+00]	 2.1682215e-01


.. parsed-literal::

      74	 1.2512873e+00	 1.5039726e-01	 1.3019540e+00	 1.7265882e-01	[ 1.2455422e+00]	 2.0757866e-01
      75	 1.2569366e+00	 1.5029056e-01	 1.3078840e+00	 1.7254408e-01	[ 1.2497327e+00]	 1.9277644e-01


.. parsed-literal::

      76	 1.2654817e+00	 1.5031114e-01	 1.3166298e+00	 1.7299417e-01	[ 1.2551669e+00]	 2.1731472e-01


.. parsed-literal::

      77	 1.2702032e+00	 1.5005892e-01	 1.3220078e+00	 1.7276510e-01	  1.2542130e+00 	 2.2109914e-01


.. parsed-literal::

      78	 1.2771831e+00	 1.4978213e-01	 1.3284371e+00	 1.7214396e-01	[ 1.2624259e+00]	 2.0949626e-01
      79	 1.2814532e+00	 1.4956668e-01	 1.3325976e+00	 1.7158778e-01	[ 1.2667867e+00]	 1.9119883e-01


.. parsed-literal::

      80	 1.2866681e+00	 1.4925762e-01	 1.3379261e+00	 1.7018708e-01	[ 1.2681056e+00]	 2.0446396e-01


.. parsed-literal::

      81	 1.2924382e+00	 1.4882885e-01	 1.3437189e+00	 1.6904627e-01	[ 1.2708308e+00]	 2.0367718e-01


.. parsed-literal::

      82	 1.2978279e+00	 1.4852250e-01	 1.3492277e+00	 1.6822606e-01	[ 1.2731980e+00]	 2.1048284e-01


.. parsed-literal::

      83	 1.3037868e+00	 1.4779772e-01	 1.3553524e+00	 1.6755967e-01	[ 1.2804915e+00]	 2.0539951e-01


.. parsed-literal::

      84	 1.3096925e+00	 1.4791720e-01	 1.3612784e+00	 1.6871096e-01	[ 1.2869127e+00]	 2.1067643e-01


.. parsed-literal::

      85	 1.3131360e+00	 1.4777093e-01	 1.3647556e+00	 1.6926712e-01	[ 1.2912337e+00]	 2.0704627e-01
      86	 1.3206748e+00	 1.4704370e-01	 1.3727035e+00	 1.7021972e-01	[ 1.2979041e+00]	 1.9930625e-01


.. parsed-literal::

      87	 1.3244507e+00	 1.4639128e-01	 1.3769937e+00	 1.7029309e-01	  1.2974319e+00 	 2.2494006e-01
      88	 1.3310097e+00	 1.4605584e-01	 1.3833705e+00	 1.7000183e-01	[ 1.3023394e+00]	 1.8358874e-01


.. parsed-literal::

      89	 1.3354620e+00	 1.4564544e-01	 1.3878845e+00	 1.6937375e-01	[ 1.3034591e+00]	 2.0578742e-01


.. parsed-literal::

      90	 1.3392698e+00	 1.4525772e-01	 1.3917939e+00	 1.6896211e-01	[ 1.3034860e+00]	 2.0528698e-01


.. parsed-literal::

      91	 1.3441482e+00	 1.4444732e-01	 1.3969593e+00	 1.6835533e-01	[ 1.3043131e+00]	 2.1082282e-01


.. parsed-literal::

      92	 1.3488272e+00	 1.4414854e-01	 1.4017012e+00	 1.6839138e-01	  1.3034372e+00 	 2.1365261e-01


.. parsed-literal::

      93	 1.3521971e+00	 1.4387690e-01	 1.4050322e+00	 1.6833435e-01	[ 1.3059354e+00]	 2.0721817e-01


.. parsed-literal::

      94	 1.3575131e+00	 1.4333099e-01	 1.4104622e+00	 1.6831127e-01	[ 1.3104430e+00]	 2.1411967e-01


.. parsed-literal::

      95	 1.3614815e+00	 1.4289887e-01	 1.4144132e+00	 1.6711182e-01	  1.3080899e+00 	 2.1188617e-01
      96	 1.3655584e+00	 1.4249805e-01	 1.4184730e+00	 1.6649580e-01	  1.3094064e+00 	 1.8529677e-01


.. parsed-literal::

      97	 1.3712424e+00	 1.4169395e-01	 1.4241982e+00	 1.6540333e-01	  1.3094827e+00 	 2.0180869e-01


.. parsed-literal::

      98	 1.3741340e+00	 1.4128671e-01	 1.4271067e+00	 1.6522032e-01	  1.3097136e+00 	 2.0818639e-01
      99	 1.3775216e+00	 1.4087022e-01	 1.4304535e+00	 1.6482318e-01	[ 1.3129950e+00]	 2.0466495e-01


.. parsed-literal::

     100	 1.3817052e+00	 1.4015019e-01	 1.4347164e+00	 1.6474732e-01	[ 1.3143820e+00]	 2.1466327e-01


.. parsed-literal::

     101	 1.3854786e+00	 1.3950718e-01	 1.4385340e+00	 1.6459878e-01	[ 1.3180802e+00]	 2.0798063e-01
     102	 1.3889806e+00	 1.3886438e-01	 1.4420678e+00	 1.6447160e-01	[ 1.3187000e+00]	 1.9591784e-01


.. parsed-literal::

     103	 1.3916626e+00	 1.3825027e-01	 1.4448002e+00	 1.6386492e-01	[ 1.3196231e+00]	 2.0924902e-01


.. parsed-literal::

     104	 1.3949109e+00	 1.3766616e-01	 1.4480827e+00	 1.6295551e-01	[ 1.3208492e+00]	 2.1831989e-01
     105	 1.3983416e+00	 1.3718387e-01	 1.4515609e+00	 1.6212009e-01	[ 1.3208597e+00]	 1.9616890e-01


.. parsed-literal::

     106	 1.4021806e+00	 1.3671109e-01	 1.4554698e+00	 1.6165810e-01	  1.3200572e+00 	 2.0127511e-01


.. parsed-literal::

     107	 1.4055555e+00	 1.3610992e-01	 1.4589721e+00	 1.6091167e-01	  1.3131894e+00 	 2.1283770e-01
     108	 1.4084536e+00	 1.3597782e-01	 1.4618657e+00	 1.6119167e-01	  1.3116408e+00 	 1.9958854e-01


.. parsed-literal::

     109	 1.4113208e+00	 1.3577827e-01	 1.4647371e+00	 1.6128527e-01	  1.3088465e+00 	 2.0645761e-01
     110	 1.4154295e+00	 1.3501858e-01	 1.4689854e+00	 1.6079284e-01	  1.3008426e+00 	 1.9763231e-01


.. parsed-literal::

     111	 1.4172929e+00	 1.3463817e-01	 1.4710248e+00	 1.6030312e-01	  1.2894944e+00 	 2.0713449e-01


.. parsed-literal::

     112	 1.4210267e+00	 1.3423930e-01	 1.4746560e+00	 1.5975119e-01	  1.2934551e+00 	 2.1815872e-01


.. parsed-literal::

     113	 1.4232307e+00	 1.3391464e-01	 1.4768580e+00	 1.5926473e-01	  1.2965677e+00 	 2.0765090e-01


.. parsed-literal::

     114	 1.4263949e+00	 1.3354839e-01	 1.4800784e+00	 1.5874350e-01	  1.3006128e+00 	 2.1309400e-01
     115	 1.4289876e+00	 1.3300167e-01	 1.4829278e+00	 1.5846163e-01	  1.3010169e+00 	 1.7306733e-01


.. parsed-literal::

     116	 1.4322627e+00	 1.3288198e-01	 1.4861132e+00	 1.5824823e-01	  1.3034094e+00 	 1.9771552e-01


.. parsed-literal::

     117	 1.4346954e+00	 1.3265582e-01	 1.4885760e+00	 1.5811167e-01	  1.3013184e+00 	 2.2119284e-01


.. parsed-literal::

     118	 1.4377368e+00	 1.3228478e-01	 1.4917021e+00	 1.5788804e-01	  1.2988547e+00 	 2.0812297e-01
     119	 1.4390287e+00	 1.3174776e-01	 1.4932585e+00	 1.5735699e-01	  1.2901894e+00 	 1.9549346e-01


.. parsed-literal::

     120	 1.4430979e+00	 1.3162868e-01	 1.4971844e+00	 1.5741024e-01	  1.2943012e+00 	 2.1377182e-01
     121	 1.4445268e+00	 1.3150766e-01	 1.4985890e+00	 1.5725969e-01	  1.2956101e+00 	 1.9803333e-01


.. parsed-literal::

     122	 1.4466872e+00	 1.3128291e-01	 1.5007599e+00	 1.5708215e-01	  1.2957783e+00 	 1.9441319e-01


.. parsed-literal::

     123	 1.4478564e+00	 1.3091791e-01	 1.5020109e+00	 1.5639484e-01	  1.2900252e+00 	 2.1529675e-01


.. parsed-literal::

     124	 1.4506942e+00	 1.3082103e-01	 1.5048366e+00	 1.5652563e-01	  1.2912242e+00 	 2.1142244e-01
     125	 1.4526002e+00	 1.3073223e-01	 1.5067755e+00	 1.5659221e-01	  1.2900017e+00 	 1.9643116e-01


.. parsed-literal::

     126	 1.4544524e+00	 1.3059604e-01	 1.5086878e+00	 1.5659184e-01	  1.2875227e+00 	 2.0947313e-01
     127	 1.4573141e+00	 1.3037544e-01	 1.5116195e+00	 1.5656363e-01	  1.2839648e+00 	 1.9708824e-01


.. parsed-literal::

     128	 1.4583315e+00	 1.2979969e-01	 1.5128886e+00	 1.5621321e-01	  1.2803158e+00 	 2.0676827e-01


.. parsed-literal::

     129	 1.4630397e+00	 1.2968385e-01	 1.5174646e+00	 1.5589821e-01	  1.2804458e+00 	 2.0380163e-01
     130	 1.4646206e+00	 1.2960427e-01	 1.5189874e+00	 1.5571729e-01	  1.2833494e+00 	 1.8405700e-01


.. parsed-literal::

     131	 1.4672365e+00	 1.2929254e-01	 1.5216434e+00	 1.5525536e-01	  1.2851998e+00 	 1.8041396e-01
     132	 1.4692910e+00	 1.2888256e-01	 1.5238188e+00	 1.5490467e-01	  1.2888296e+00 	 1.8788958e-01


.. parsed-literal::

     133	 1.4719454e+00	 1.2866999e-01	 1.5264777e+00	 1.5476433e-01	  1.2895962e+00 	 2.0820069e-01


.. parsed-literal::

     134	 1.4742332e+00	 1.2840488e-01	 1.5288550e+00	 1.5465647e-01	  1.2867392e+00 	 2.0794868e-01
     135	 1.4761492e+00	 1.2816811e-01	 1.5308653e+00	 1.5475283e-01	  1.2843286e+00 	 1.8509865e-01


.. parsed-literal::

     136	 1.4776826e+00	 1.2774455e-01	 1.5327451e+00	 1.5473207e-01	  1.2738196e+00 	 2.0205688e-01


.. parsed-literal::

     137	 1.4802570e+00	 1.2764492e-01	 1.5352047e+00	 1.5501444e-01	  1.2761431e+00 	 2.2032714e-01
     138	 1.4813763e+00	 1.2761293e-01	 1.5362605e+00	 1.5514701e-01	  1.2775684e+00 	 1.9424415e-01


.. parsed-literal::

     139	 1.4831358e+00	 1.2744116e-01	 1.5380040e+00	 1.5520883e-01	  1.2777882e+00 	 1.7868328e-01
     140	 1.4850244e+00	 1.2706150e-01	 1.5399291e+00	 1.5508454e-01	  1.2788144e+00 	 1.9540858e-01


.. parsed-literal::

     141	 1.4876116e+00	 1.2686050e-01	 1.5425024e+00	 1.5528479e-01	  1.2778236e+00 	 2.0601106e-01


.. parsed-literal::

     142	 1.4887377e+00	 1.2673637e-01	 1.5436565e+00	 1.5519562e-01	  1.2763992e+00 	 2.2627044e-01


.. parsed-literal::

     143	 1.4905205e+00	 1.2655947e-01	 1.5455093e+00	 1.5517993e-01	  1.2719755e+00 	 2.0683217e-01


.. parsed-literal::

     144	 1.4927479e+00	 1.2634777e-01	 1.5478943e+00	 1.5530966e-01	  1.2578162e+00 	 2.0924687e-01


.. parsed-literal::

     145	 1.4949830e+00	 1.2618724e-01	 1.5502083e+00	 1.5542444e-01	  1.2503648e+00 	 2.1208096e-01


.. parsed-literal::

     146	 1.4968226e+00	 1.2598577e-01	 1.5520956e+00	 1.5553507e-01	  1.2437241e+00 	 2.0465589e-01


.. parsed-literal::

     147	 1.4986525e+00	 1.2573541e-01	 1.5539592e+00	 1.5544081e-01	  1.2370862e+00 	 2.2451162e-01
     148	 1.5001282e+00	 1.2554767e-01	 1.5554017e+00	 1.5546791e-01	  1.2343493e+00 	 1.7501378e-01


.. parsed-literal::

     149	 1.5014681e+00	 1.2543452e-01	 1.5566781e+00	 1.5526053e-01	  1.2356632e+00 	 2.0148134e-01


.. parsed-literal::

     150	 1.5029967e+00	 1.2524435e-01	 1.5581819e+00	 1.5508134e-01	  1.2353961e+00 	 2.1093011e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 2s, sys: 946 ms, total: 2min 3s
    Wall time: 31.1 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f0cd0a24a60>



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
    CPU times: user 1.69 s, sys: 35 ms, total: 1.72 s
    Wall time: 529 ms


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

