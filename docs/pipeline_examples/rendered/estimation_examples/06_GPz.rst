GPz Estimation Example
======================

**Author:** Sam Schmidt

**Last Run Successfully:** September 26, 2023

**Note:** If you’re planning to run this in a notebook, you may want to
use interactive mode instead. See
`GPz.ipynb <https://github.com/LSSTDESC/rail/blob/main/interactive_examples/estimation_examples/GPz.ipynb>`__
in the ``interactive_examples/estimation_examples/`` folder for a
version of this notebook in interactive mode.

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
    import tables_io
    import qp
    from rail.core.data import TableHandle
    from rail.core.stage import RailStage
    from rail.estimation.algos.gpz import GPzInformer, GPzEstimator

.. code:: ipython3

    # find_rail_file is a convenience function that finds a file in the RAIL ecosystem   We have several example data files that are copied with RAIL that we can use for our example run, let's grab those files, one for training/validation, and the other for testing:
    from rail.utils.path_utils import find_rail_file
    trainFile = find_rail_file('examples_data/testdata/test_dc2_training_9816.hdf5')
    testFile = find_rail_file('examples_data/testdata/test_dc2_validation_9816.hdf5')
    training_data = tables_io.read(trainFile)
    test_data = tables_io.read(testFile)

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
       1	-3.4325127e-01	 3.2044902e-01	-3.3296693e-01	 3.2070579e-01	[-3.3306392e-01]	 4.4490480e-01


.. parsed-literal::

       2	-2.7445530e-01	 3.1045537e-01	-2.5023110e-01	 3.1224007e-01	[-2.5368868e-01]	 2.2114587e-01


.. parsed-literal::

       3	-2.3003372e-01	 2.8862529e-01	-1.8569179e-01	 2.9340830e-01	[-2.0384029e-01]	 2.6888490e-01
       4	-1.8775819e-01	 2.6578433e-01	-1.4364188e-01	 2.7096603e-01	[-1.7136099e-01]	 1.8255949e-01


.. parsed-literal::

       5	-1.0030477e-01	 2.5658731e-01	-6.8414520e-02	 2.5984615e-01	[-7.8270346e-02]	 1.9873381e-01
       6	-6.8229305e-02	 2.5142877e-01	-3.9926658e-02	 2.5185020e-01	[-4.1831883e-02]	 1.7977333e-01


.. parsed-literal::

       7	-5.1130352e-02	 2.4873705e-01	-2.7977053e-02	 2.4947176e-01	[-2.9898954e-02]	 1.9220591e-01


.. parsed-literal::

       8	-3.7262733e-02	 2.4626283e-01	-1.7625229e-02	 2.4715755e-01	[-1.9560930e-02]	 2.1242261e-01
       9	-2.6060641e-02	 2.4412173e-01	-8.5379303e-03	 2.4501348e-01	[-1.1019109e-02]	 1.9603896e-01


.. parsed-literal::

      10	-1.6073367e-02	 2.4215459e-01	-4.3779619e-05	 2.4365124e-01	[-5.1955185e-03]	 2.0395517e-01
      11	-1.1647994e-02	 2.4134163e-01	 3.3709942e-03	 2.4234771e-01	[-7.9826917e-04]	 1.8457484e-01


.. parsed-literal::

      12	-8.3841543e-03	 2.4077834e-01	 6.5618659e-03	 2.4175434e-01	[ 2.9089973e-03]	 2.0558333e-01
      13	-4.9362405e-03	 2.4007702e-01	 9.9258804e-03	 2.4104043e-01	[ 6.5344816e-03]	 1.9894028e-01


.. parsed-literal::

      14	 8.5973227e-04	 2.3871160e-01	 1.6541719e-02	 2.3974063e-01	[ 1.2923706e-02]	 2.0915389e-01


.. parsed-literal::

      15	 6.4763235e-02	 2.2552101e-01	 8.3141091e-02	 2.3015631e-01	[ 7.5228827e-02]	 2.0385981e-01
      16	 8.5090265e-02	 2.2353190e-01	 1.0689698e-01	 2.2781321e-01	[ 1.0340348e-01]	 1.9929433e-01


.. parsed-literal::

      17	 1.6911569e-01	 2.1832751e-01	 1.9192873e-01	 2.1930299e-01	[ 1.8845632e-01]	 1.8007135e-01
      18	 2.7077688e-01	 2.1617821e-01	 3.0253710e-01	 2.1580607e-01	[ 2.9659685e-01]	 1.9951677e-01


.. parsed-literal::

      19	 3.1539172e-01	 2.1274113e-01	 3.4777685e-01	 2.1368250e-01	[ 3.3962226e-01]	 2.1609831e-01


.. parsed-literal::

      20	 3.6066066e-01	 2.0943618e-01	 3.9253784e-01	 2.1105932e-01	[ 3.7774981e-01]	 2.0893478e-01


.. parsed-literal::

      21	 4.3576435e-01	 2.0726855e-01	 4.6856036e-01	 2.1101980e-01	[ 4.4140506e-01]	 2.1059966e-01
      22	 5.3491722e-01	 2.0804639e-01	 5.7061751e-01	 2.1246243e-01	[ 5.1267526e-01]	 1.8330550e-01


.. parsed-literal::

      23	 5.8480886e-01	 2.0842259e-01	 6.2643318e-01	 2.1084115e-01	  4.6923378e-01 	 2.0542121e-01


.. parsed-literal::

      24	 6.3523411e-01	 2.0470292e-01	 6.7419375e-01	 2.0834129e-01	[ 5.5879072e-01]	 2.0275116e-01


.. parsed-literal::

      25	 6.6287349e-01	 2.0219064e-01	 7.0118817e-01	 2.0511644e-01	[ 5.9330415e-01]	 2.0940232e-01


.. parsed-literal::

      26	 7.0122569e-01	 1.9939041e-01	 7.3856418e-01	 2.0022504e-01	[ 6.3996426e-01]	 2.9515076e-01


.. parsed-literal::

      27	 7.2974320e-01	 1.9808874e-01	 7.6759695e-01	 1.9919496e-01	[ 6.8984531e-01]	 2.0678592e-01


.. parsed-literal::

      28	 7.5640815e-01	 1.9686296e-01	 7.9524778e-01	 1.9796124e-01	[ 7.0152711e-01]	 2.1340513e-01


.. parsed-literal::

      29	 7.7646116e-01	 1.9505367e-01	 8.1558759e-01	 1.9758036e-01	[ 7.3371413e-01]	 2.1304321e-01


.. parsed-literal::

      30	 7.9538480e-01	 1.9319592e-01	 8.3463961e-01	 1.9605151e-01	[ 7.5692784e-01]	 2.1092367e-01


.. parsed-literal::

      31	 8.1640255e-01	 1.9237960e-01	 8.5610869e-01	 1.9608647e-01	[ 7.7191228e-01]	 2.1697187e-01
      32	 8.4840790e-01	 1.9248075e-01	 8.8974233e-01	 1.9585631e-01	[ 8.0958640e-01]	 1.9594026e-01


.. parsed-literal::

      33	 8.6377331e-01	 1.9093988e-01	 9.0609685e-01	 1.9431237e-01	[ 8.2461828e-01]	 1.9776654e-01


.. parsed-literal::

      34	 8.7961341e-01	 1.8947380e-01	 9.2144055e-01	 1.9301684e-01	[ 8.4447250e-01]	 2.1203637e-01
      35	 8.9177097e-01	 1.8815820e-01	 9.3325256e-01	 1.9178897e-01	[ 8.5976630e-01]	 2.0571423e-01


.. parsed-literal::

      36	 9.1116811e-01	 1.8753532e-01	 9.5295854e-01	 1.9184618e-01	[ 8.7730046e-01]	 2.1059752e-01


.. parsed-literal::

      37	 9.2997159e-01	 1.8723609e-01	 9.7352565e-01	 1.9112497e-01	[ 8.9908647e-01]	 2.0392990e-01


.. parsed-literal::

      38	 9.4664964e-01	 1.8617416e-01	 9.9069084e-01	 1.9062626e-01	[ 9.1350564e-01]	 2.1286535e-01
      39	 9.5646098e-01	 1.8569195e-01	 1.0004088e+00	 1.9009903e-01	[ 9.2500104e-01]	 2.0190191e-01


.. parsed-literal::

      40	 9.7051093e-01	 1.8470180e-01	 1.0156702e+00	 1.8924173e-01	[ 9.3885156e-01]	 2.0885324e-01
      41	 9.8491705e-01	 1.8438135e-01	 1.0312025e+00	 1.8855261e-01	[ 9.5030551e-01]	 1.9541121e-01


.. parsed-literal::

      42	 9.9944009e-01	 1.8344499e-01	 1.0467247e+00	 1.8677517e-01	[ 9.6437556e-01]	 2.0981312e-01
      43	 1.0056101e+00	 1.8443731e-01	 1.0537234e+00	 1.8794800e-01	[ 9.7022269e-01]	 1.9796777e-01


.. parsed-literal::

      44	 1.0189652e+00	 1.8290167e-01	 1.0663932e+00	 1.8606905e-01	[ 9.8505006e-01]	 2.0744014e-01


.. parsed-literal::

      45	 1.0251035e+00	 1.8192132e-01	 1.0724511e+00	 1.8487355e-01	[ 9.9150977e-01]	 2.0900106e-01
      46	 1.0361229e+00	 1.7992710e-01	 1.0834850e+00	 1.8276452e-01	[ 1.0020140e+00]	 1.8095112e-01


.. parsed-literal::

      47	 1.0473027e+00	 1.7744397e-01	 1.0947794e+00	 1.8102530e-01	[ 1.0110275e+00]	 1.9852996e-01


.. parsed-literal::

      48	 1.0555444e+00	 1.7575370e-01	 1.1032306e+00	 1.7960503e-01	[ 1.0191401e+00]	 2.0930266e-01


.. parsed-literal::

      49	 1.0626419e+00	 1.7446970e-01	 1.1103566e+00	 1.7842878e-01	[ 1.0249205e+00]	 2.1745872e-01


.. parsed-literal::

      50	 1.0709449e+00	 1.7236227e-01	 1.1189759e+00	 1.7619479e-01	[ 1.0307747e+00]	 2.1445584e-01


.. parsed-literal::

      51	 1.0782388e+00	 1.7069692e-01	 1.1265327e+00	 1.7419346e-01	[ 1.0377081e+00]	 2.0710421e-01
      52	 1.0916264e+00	 1.6757845e-01	 1.1407231e+00	 1.7047100e-01	[ 1.0478441e+00]	 1.8641782e-01


.. parsed-literal::

      53	 1.0955594e+00	 1.6591226e-01	 1.1449350e+00	 1.6870130e-01	[ 1.0606020e+00]	 2.0871472e-01


.. parsed-literal::

      54	 1.1057185e+00	 1.6533529e-01	 1.1548028e+00	 1.6804793e-01	[ 1.0663662e+00]	 2.0772862e-01


.. parsed-literal::

      55	 1.1131259e+00	 1.6429118e-01	 1.1622747e+00	 1.6732838e-01	[ 1.0707516e+00]	 2.1122193e-01


.. parsed-literal::

      56	 1.1206182e+00	 1.6285866e-01	 1.1699096e+00	 1.6598906e-01	[ 1.0770913e+00]	 2.0423102e-01
      57	 1.1309078e+00	 1.6016145e-01	 1.1806956e+00	 1.6460373e-01	[ 1.0928429e+00]	 1.6881275e-01


.. parsed-literal::

      58	 1.1391969e+00	 1.5929356e-01	 1.1895033e+00	 1.6350335e-01	[ 1.0988911e+00]	 1.9063950e-01
      59	 1.1459953e+00	 1.5882034e-01	 1.1961928e+00	 1.6319830e-01	[ 1.1043125e+00]	 1.9122958e-01


.. parsed-literal::

      60	 1.1544145e+00	 1.5786853e-01	 1.2050613e+00	 1.6280408e-01	[ 1.1106709e+00]	 2.0821476e-01
      61	 1.1636628e+00	 1.5741744e-01	 1.2149074e+00	 1.6281374e-01	[ 1.1182952e+00]	 2.0022678e-01


.. parsed-literal::

      62	 1.1700996e+00	 1.5556454e-01	 1.2219768e+00	 1.6202220e-01	[ 1.1237710e+00]	 3.1350088e-01
      63	 1.1774976e+00	 1.5550653e-01	 1.2297351e+00	 1.6223302e-01	[ 1.1328930e+00]	 1.6905284e-01


.. parsed-literal::

      64	 1.1829982e+00	 1.5505209e-01	 1.2353888e+00	 1.6217926e-01	[ 1.1361486e+00]	 1.9820523e-01
      65	 1.1908004e+00	 1.5448158e-01	 1.2434334e+00	 1.6206838e-01	[ 1.1432440e+00]	 1.8083477e-01


.. parsed-literal::

      66	 1.1986603e+00	 1.5327593e-01	 1.2515789e+00	 1.6195630e-01	  1.1334976e+00 	 2.0618439e-01
      67	 1.2067164e+00	 1.5286610e-01	 1.2595451e+00	 1.6152954e-01	  1.1374816e+00 	 2.0039845e-01


.. parsed-literal::

      68	 1.2140731e+00	 1.5207067e-01	 1.2670940e+00	 1.6060238e-01	  1.1330122e+00 	 1.9995952e-01
      69	 1.2201680e+00	 1.5111747e-01	 1.2733218e+00	 1.5936696e-01	  1.1241068e+00 	 2.0349574e-01


.. parsed-literal::

      70	 1.2258490e+00	 1.5028602e-01	 1.2789386e+00	 1.5802171e-01	  1.1207815e+00 	 1.8853712e-01
      71	 1.2307553e+00	 1.4958375e-01	 1.2838799e+00	 1.5714908e-01	  1.1180430e+00 	 2.0394444e-01


.. parsed-literal::

      72	 1.2387542e+00	 1.4889966e-01	 1.2921126e+00	 1.5546280e-01	  1.1126985e+00 	 2.0261574e-01
      73	 1.2438087e+00	 1.4778906e-01	 1.2975886e+00	 1.5397514e-01	  1.0825041e+00 	 2.0102406e-01


.. parsed-literal::

      74	 1.2496798e+00	 1.4776925e-01	 1.3031683e+00	 1.5409285e-01	  1.0965378e+00 	 2.1246576e-01


.. parsed-literal::

      75	 1.2545844e+00	 1.4735738e-01	 1.3081025e+00	 1.5373561e-01	  1.0927851e+00 	 2.0828223e-01
      76	 1.2607324e+00	 1.4667248e-01	 1.3143003e+00	 1.5286411e-01	  1.0964648e+00 	 2.0785403e-01


.. parsed-literal::

      77	 1.2674480e+00	 1.4550301e-01	 1.3211952e+00	 1.5109872e-01	  1.0918743e+00 	 1.8210602e-01
      78	 1.2734411e+00	 1.4494999e-01	 1.3272737e+00	 1.5034426e-01	  1.0966431e+00 	 1.9721103e-01


.. parsed-literal::

      79	 1.2780240e+00	 1.4475649e-01	 1.3318555e+00	 1.5001374e-01	  1.1082118e+00 	 1.9725800e-01


.. parsed-literal::

      80	 1.2842671e+00	 1.4440281e-01	 1.3385522e+00	 1.4944358e-01	  1.1044640e+00 	 2.1151972e-01


.. parsed-literal::

      81	 1.2903027e+00	 1.4414952e-01	 1.3446730e+00	 1.4875688e-01	  1.1190762e+00 	 2.0471907e-01
      82	 1.2952225e+00	 1.4404534e-01	 1.3496129e+00	 1.4858817e-01	  1.1208799e+00 	 2.0444846e-01


.. parsed-literal::

      83	 1.3027564e+00	 1.4408335e-01	 1.3574455e+00	 1.4833642e-01	  1.1195763e+00 	 2.0314074e-01
      84	 1.3059210e+00	 1.4437441e-01	 1.3608290e+00	 1.4855370e-01	  1.1190640e+00 	 2.0191574e-01


.. parsed-literal::

      85	 1.3100139e+00	 1.4439805e-01	 1.3646766e+00	 1.4871804e-01	  1.1225234e+00 	 2.1052766e-01


.. parsed-literal::

      86	 1.3134701e+00	 1.4407374e-01	 1.3682648e+00	 1.4819963e-01	  1.1229946e+00 	 2.0862031e-01
      87	 1.3169579e+00	 1.4383031e-01	 1.3718478e+00	 1.4801352e-01	  1.1216323e+00 	 2.0464540e-01


.. parsed-literal::

      88	 1.3242949e+00	 1.4309714e-01	 1.3793506e+00	 1.4764534e-01	  1.1098295e+00 	 2.1208763e-01


.. parsed-literal::

      89	 1.3286723e+00	 1.4252815e-01	 1.3839830e+00	 1.4720346e-01	  1.1034551e+00 	 3.2584596e-01


.. parsed-literal::

      90	 1.3323124e+00	 1.4229100e-01	 1.3876047e+00	 1.4713578e-01	  1.1029818e+00 	 2.1025515e-01
      91	 1.3364079e+00	 1.4200612e-01	 1.3918085e+00	 1.4681144e-01	  1.1051426e+00 	 1.9456625e-01


.. parsed-literal::

      92	 1.3412034e+00	 1.4133507e-01	 1.3967763e+00	 1.4602986e-01	  1.1119887e+00 	 1.9702077e-01


.. parsed-literal::

      93	 1.3461440e+00	 1.4066059e-01	 1.4021033e+00	 1.4525196e-01	  1.1196478e+00 	 2.0185900e-01
      94	 1.3498907e+00	 1.4041442e-01	 1.4058905e+00	 1.4474297e-01	  1.1239355e+00 	 2.0575523e-01


.. parsed-literal::

      95	 1.3539950e+00	 1.4013964e-01	 1.4100396e+00	 1.4442586e-01	  1.1271403e+00 	 2.0415974e-01
      96	 1.3575711e+00	 1.3960737e-01	 1.4139106e+00	 1.4399868e-01	  1.1268886e+00 	 1.7785120e-01


.. parsed-literal::

      97	 1.3615996e+00	 1.3950678e-01	 1.4179946e+00	 1.4384282e-01	  1.1332159e+00 	 2.1015406e-01


.. parsed-literal::

      98	 1.3650593e+00	 1.3941812e-01	 1.4214290e+00	 1.4355277e-01	  1.1386462e+00 	 2.1264839e-01


.. parsed-literal::

      99	 1.3681158e+00	 1.3939349e-01	 1.4245709e+00	 1.4346745e-01	[ 1.1480752e+00]	 2.1402311e-01
     100	 1.3724994e+00	 1.3903112e-01	 1.4288487e+00	 1.4290807e-01	[ 1.1564645e+00]	 1.9487095e-01


.. parsed-literal::

     101	 1.3757751e+00	 1.3874249e-01	 1.4320347e+00	 1.4259089e-01	[ 1.1671250e+00]	 2.1472549e-01


.. parsed-literal::

     102	 1.3789219e+00	 1.3821983e-01	 1.4352106e+00	 1.4214751e-01	[ 1.1734319e+00]	 2.1322370e-01


.. parsed-literal::

     103	 1.3829364e+00	 1.3738114e-01	 1.4392961e+00	 1.4131326e-01	[ 1.1792303e+00]	 2.0882344e-01
     104	 1.3868506e+00	 1.3690142e-01	 1.4433830e+00	 1.4093464e-01	[ 1.1826403e+00]	 1.7030334e-01


.. parsed-literal::

     105	 1.3900135e+00	 1.3646690e-01	 1.4465256e+00	 1.4048336e-01	[ 1.1833235e+00]	 1.8886590e-01
     106	 1.3930897e+00	 1.3615415e-01	 1.4496607e+00	 1.4008136e-01	[ 1.1857384e+00]	 1.9256496e-01


.. parsed-literal::

     107	 1.3966519e+00	 1.3549897e-01	 1.4533225e+00	 1.3948750e-01	[ 1.1902598e+00]	 1.9752955e-01
     108	 1.4006174e+00	 1.3457114e-01	 1.4574345e+00	 1.3878144e-01	[ 1.1943768e+00]	 1.9820237e-01


.. parsed-literal::

     109	 1.4039688e+00	 1.3414793e-01	 1.4607993e+00	 1.3858635e-01	[ 1.2034745e+00]	 2.0615840e-01
     110	 1.4067097e+00	 1.3383718e-01	 1.4635273e+00	 1.3845717e-01	[ 1.2075733e+00]	 1.7615008e-01


.. parsed-literal::

     111	 1.4113692e+00	 1.3294151e-01	 1.4682976e+00	 1.3770973e-01	[ 1.2121456e+00]	 2.0914292e-01
     112	 1.4134240e+00	 1.3211017e-01	 1.4705505e+00	 1.3724967e-01	[ 1.2197134e+00]	 2.0155501e-01


.. parsed-literal::

     113	 1.4180318e+00	 1.3180795e-01	 1.4750409e+00	 1.3670768e-01	  1.2180883e+00 	 1.9758177e-01
     114	 1.4199241e+00	 1.3173487e-01	 1.4768671e+00	 1.3649326e-01	  1.2188619e+00 	 1.9428062e-01


.. parsed-literal::

     115	 1.4227755e+00	 1.3138587e-01	 1.4797778e+00	 1.3606196e-01	  1.2185327e+00 	 1.9677281e-01
     116	 1.4252374e+00	 1.3074381e-01	 1.4825449e+00	 1.3555285e-01	  1.2136010e+00 	 1.9692183e-01


.. parsed-literal::

     117	 1.4285178e+00	 1.3062576e-01	 1.4858194e+00	 1.3560225e-01	  1.2147534e+00 	 2.0223832e-01
     118	 1.4303416e+00	 1.3038596e-01	 1.4876652e+00	 1.3562272e-01	  1.2176562e+00 	 1.9848156e-01


.. parsed-literal::

     119	 1.4326812e+00	 1.3013133e-01	 1.4900788e+00	 1.3561614e-01	  1.2193360e+00 	 1.8460941e-01


.. parsed-literal::

     120	 1.4358640e+00	 1.2993088e-01	 1.4934494e+00	 1.3570565e-01	  1.2186655e+00 	 2.1106577e-01


.. parsed-literal::

     121	 1.4384833e+00	 1.2967827e-01	 1.4961831e+00	 1.3547809e-01	[ 1.2228591e+00]	 2.0196819e-01
     122	 1.4405983e+00	 1.2960688e-01	 1.4982015e+00	 1.3524260e-01	  1.2228548e+00 	 2.0294023e-01


.. parsed-literal::

     123	 1.4432205e+00	 1.2947833e-01	 1.5007706e+00	 1.3491312e-01	[ 1.2234317e+00]	 2.1512222e-01


.. parsed-literal::

     124	 1.4453410e+00	 1.2914731e-01	 1.5028790e+00	 1.3459034e-01	  1.2232988e+00 	 2.0594144e-01
     125	 1.4480507e+00	 1.2880885e-01	 1.5056478e+00	 1.3428652e-01	[ 1.2290752e+00]	 1.8622231e-01


.. parsed-literal::

     126	 1.4505172e+00	 1.2840417e-01	 1.5082055e+00	 1.3411191e-01	[ 1.2315184e+00]	 2.0954204e-01


.. parsed-literal::

     127	 1.4529779e+00	 1.2802793e-01	 1.5107821e+00	 1.3406090e-01	[ 1.2359641e+00]	 2.0461130e-01
     128	 1.4543638e+00	 1.2789965e-01	 1.5123130e+00	 1.3382151e-01	[ 1.2391567e+00]	 1.7845607e-01


.. parsed-literal::

     129	 1.4563874e+00	 1.2787287e-01	 1.5142559e+00	 1.3383502e-01	  1.2381232e+00 	 1.9536996e-01


.. parsed-literal::

     130	 1.4581476e+00	 1.2784742e-01	 1.5160012e+00	 1.3367521e-01	  1.2370798e+00 	 2.0526338e-01
     131	 1.4595799e+00	 1.2778053e-01	 1.5173988e+00	 1.3351618e-01	  1.2378819e+00 	 1.8289137e-01


.. parsed-literal::

     132	 1.4629008e+00	 1.2752841e-01	 1.5206375e+00	 1.3320386e-01	[ 1.2414398e+00]	 2.0958447e-01


.. parsed-literal::

     133	 1.4648228e+00	 1.2731916e-01	 1.5225478e+00	 1.3300646e-01	[ 1.2423284e+00]	 3.2693362e-01


.. parsed-literal::

     134	 1.4668877e+00	 1.2715002e-01	 1.5245576e+00	 1.3294758e-01	[ 1.2459977e+00]	 2.1689057e-01


.. parsed-literal::

     135	 1.4690091e+00	 1.2686197e-01	 1.5267355e+00	 1.3294941e-01	[ 1.2465477e+00]	 2.1262097e-01
     136	 1.4707768e+00	 1.2664566e-01	 1.5286304e+00	 1.3289148e-01	[ 1.2476678e+00]	 1.7921352e-01


.. parsed-literal::

     137	 1.4727976e+00	 1.2636819e-01	 1.5307835e+00	 1.3298811e-01	  1.2426937e+00 	 2.0370269e-01


.. parsed-literal::

     138	 1.4745522e+00	 1.2620749e-01	 1.5326226e+00	 1.3301439e-01	  1.2411192e+00 	 2.0111609e-01


.. parsed-literal::

     139	 1.4769950e+00	 1.2599392e-01	 1.5351953e+00	 1.3313539e-01	  1.2392877e+00 	 2.0265555e-01


.. parsed-literal::

     140	 1.4793162e+00	 1.2570985e-01	 1.5375883e+00	 1.3334367e-01	  1.2404113e+00 	 2.0176792e-01
     141	 1.4814718e+00	 1.2553143e-01	 1.5396378e+00	 1.3340202e-01	  1.2423310e+00 	 2.0052719e-01


.. parsed-literal::

     142	 1.4834043e+00	 1.2524306e-01	 1.5414574e+00	 1.3348484e-01	  1.2436917e+00 	 2.0341063e-01


.. parsed-literal::

     143	 1.4851222e+00	 1.2494034e-01	 1.5431245e+00	 1.3348932e-01	  1.2433497e+00 	 2.1153593e-01
     144	 1.4865529e+00	 1.2434372e-01	 1.5446610e+00	 1.3389126e-01	  1.2385923e+00 	 1.9276953e-01


.. parsed-literal::

     145	 1.4890911e+00	 1.2424095e-01	 1.5471723e+00	 1.3372150e-01	  1.2394228e+00 	 1.9531751e-01
     146	 1.4902719e+00	 1.2420495e-01	 1.5484195e+00	 1.3367972e-01	  1.2380521e+00 	 1.9636607e-01


.. parsed-literal::

     147	 1.4922594e+00	 1.2403096e-01	 1.5505860e+00	 1.3373735e-01	  1.2339241e+00 	 1.9274998e-01


.. parsed-literal::

     148	 1.4936876e+00	 1.2387285e-01	 1.5521414e+00	 1.3366443e-01	  1.2292696e+00 	 3.1139612e-01


.. parsed-literal::

     149	 1.4954971e+00	 1.2364976e-01	 1.5540417e+00	 1.3385181e-01	  1.2241577e+00 	 2.1294475e-01


.. parsed-literal::

     150	 1.4970202e+00	 1.2347160e-01	 1.5555617e+00	 1.3386344e-01	  1.2246417e+00 	 2.1064305e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 3s, sys: 1.06 s, total: 2min 4s
    Wall time: 31.3 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f8214f8b580>



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

    Inserting handle into data store.  input: None, gpz_run
    Inserting handle into data store.  model: GPz_model.pkl, gpz_run
    Process 0 running estimator on chunk 0 - 20,449
    Process 0 estimating GPz PZ PDF for rows 0 - 20,449


.. parsed-literal::

    Inserting handle into data store.  output_gpz_run: inprogress_output_gpz_run.hdf5, gpz_run
    CPU times: user 957 ms, sys: 46 ms, total: 1 s
    Wall time: 365 ms


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




.. image:: 06_GPz_files/06_GPz_15_1.png


GPzEstimator parameterizes each PDF as a single Gaussian, here we see a
few examples of Gaussians of different widths. Now let’s grab the mode
of each PDF (stored as ancil data in the ensemble) and compare to the
true redshifts from the test_data file:

.. code:: ipython3

    truez = test_data['photometry']['redshift']
    zmode = ens.ancil['zmode'].flatten()

.. code:: ipython3

    plt.figure(figsize=(12,12))
    plt.scatter(truez, zmode, s=3)
    plt.plot([0,3],[0,3], 'k--')
    plt.xlabel("redshift", fontsize=12)
    plt.ylabel("z mode", fontsize=12)




.. parsed-literal::

    Text(0, 0.5, 'z mode')




.. image:: 06_GPz_files/06_GPz_18_1.png

