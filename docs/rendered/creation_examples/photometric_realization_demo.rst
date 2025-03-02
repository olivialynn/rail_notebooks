Photometric Realization from Different Magnitude Error Models
=============================================================

author: John Franklin Crenshaw, Sam Schmidt, Eric Charles, Ziang Yan

last run successfully: August 2, 2023

This notebook demonstrates how to do photometric realization from
different magnitude error models. For more completed degrader demo, see
``degradation-demo.ipynb``

.. code:: ipython3

    import matplotlib.pyplot as plt
    from pzflow.examples import get_example_flow
    from rail.creation.engines.flowEngine import FlowCreator
    from rail.creation.degraders.photometric_errors import LSSTErrorModel
    from rail.core.stage import RailStage


Specify the path to the pretrained ‘pzflow’ used to generate samples

.. code:: ipython3

    import pzflow
    import os
    
    flow_file = os.path.join(
        os.path.dirname(pzflow.__file__), "example_files", "example-flow.pzflow.pkl"
    )


We’ll start by setting up the RAIL data store. RAIL uses
`ceci <https://github.com/LSSTDESC/ceci>`__, which is designed for
pipelines rather than interactive notebooks, the data store will work
around that and enable us to use data interactively. See the
``rail/examples/goldenspike_examples/goldenspike.ipynb`` example
notebook for more details on the Data Store.

.. code:: ipython3

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True


“True” Engine
~~~~~~~~~~~~~

First, let’s make an Engine that has no degradation. We can use it to
generate a “true” sample, to which we can compare all the degraded
samples below.

Note: in this example, we will use a normalizing flow engine from the
`pzflow <https://github.com/jfcrenshaw/pzflow>`__ package. However,
everything in this notebook is totally agnostic to what the underlying
engine is.

The Engine is a type of RailStage object, so we can make one using the
``RailStage.make_stage`` function for the class of Engine that we want.
We then pass in the configuration parameters as arguments to
``make_stage``.

.. code:: ipython3

    n_samples = int(1e5)
    flowEngine_truth = FlowCreator.make_stage(
        name="truth", model=flow_file, n_samples=n_samples
    )



.. parsed-literal::

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.16/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7fa23a9a03d0>



Now we invoke the ``sample`` method to generate some samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that this will return a ``DataHandle`` object, which can keep both
the data itself, and also the path to where the data is written. When
talking to rail stages we can use this as though it were the underlying
data and pass it as an argument. This allows the rail stages to keep
track of where their inputs are coming from.

To calculate magnitude error for extended sources, we need the
information about major and minor axes of each galaxy. Here we simply
generate random values

.. code:: ipython3

    samples_truth = flowEngine_truth.sample(n_samples, seed=0)
    
    import numpy as np
    
    samples_truth.data["major"] = np.abs(
        np.random.normal(loc=0.01, scale=0.1, size=n_samples)
    )  # add major and minor axes
    b_to_a = 1 - 0.5 * np.random.rand(n_samples)
    samples_truth.data["minor"] = samples_truth.data["major"] * b_to_a
    
    print(samples_truth())
    print("Data was written to ", samples_truth.path)



.. parsed-literal::

    Inserting handle into data store.  output_truth: inprogress_output_truth.pq, truth
           redshift          u          g          r          i          z  \
    0      1.398944  27.667536  26.723337  26.032637  25.178587  24.695955   
    1      2.285624  28.786999  27.476589  26.640175  26.259745  25.865673   
    2      1.495132  30.011349  29.789337  28.200390  26.014826  25.030174   
    3      0.842594  29.306244  28.721798  27.353018  26.256907  25.529823   
    4      1.588960  26.273870  26.115387  25.950441  25.687405  25.466606   
    ...         ...        ...        ...        ...        ...        ...   
    99995  0.389450  27.270800  26.371506  25.436853  25.077412  24.852779   
    99996  1.481047  27.478113  26.735254  26.042776  25.204935  24.825092   
    99997  2.023548  26.990147  26.714737  26.377949  26.250343  25.917370   
    99998  1.548204  26.367432  26.206884  26.087980  25.876932  25.715893   
    99999  1.739491  26.881983  26.773064  26.553123  26.319622  25.955982   
    
                   y     major     minor  
    0      23.994413  0.003319  0.002869  
    1      25.391064  0.008733  0.007945  
    2      24.304707  0.103938  0.052162  
    3      25.291103  0.147522  0.143359  
    4      25.096743  0.010929  0.009473  
    ...          ...       ...       ...  
    99995  24.737946  0.086491  0.071701  
    99996  24.224169  0.044537  0.022302  
    99997  25.613836  0.073146  0.047825  
    99998  25.274899  0.100551  0.094662  
    99999  25.699642  0.059611  0.049181  
    
    [100000 rows x 9 columns]
    Data was written to  output_truth.pq


LSSTErrorModel
~~~~~~~~~~~~~~

Now, we will demonstrate the ``LSSTErrorModel``, which adds photometric
errors using a model similar to the model from `Ivezic et
al. 2019 <https://arxiv.org/abs/0805.2366>`__ (specifically, it uses the
model from this paper, without making the high SNR assumption. To
restore this assumption and therefore use the exact model from the
paper, set ``highSNR=True``.)

Let’s create an error model with the default settings for point sources:

.. code:: ipython3

    errorModel = LSSTErrorModel.make_stage(name="error_model")


For extended sources:

.. code:: ipython3

    errorModel_auto = LSSTErrorModel.make_stage(
        name="error_model_auto", extendedType="auto"
    )


.. code:: ipython3

    errorModel_gaap = LSSTErrorModel.make_stage(
        name="error_model_gaap", extendedType="gaap"
    )


Now let’s add this error model as a degrader and draw some samples with
photometric errors.

.. code:: ipython3

    samples_w_errs = errorModel(samples_truth)
    samples_w_errs()



.. parsed-literal::

    Inserting handle into data store.  output_error_model: inprogress_output_error_model.pq, error_model




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398944</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.892702</td>
          <td>0.193547</td>
          <td>26.078393</td>
          <td>0.084491</td>
          <td>25.085625</td>
          <td>0.057279</td>
          <td>24.766994</td>
          <td>0.082662</td>
          <td>24.190243</td>
          <td>0.111632</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.319218</td>
          <td>0.275502</td>
          <td>26.682912</td>
          <td>0.143146</td>
          <td>26.105086</td>
          <td>0.140187</td>
          <td>25.636675</td>
          <td>0.175791</td>
          <td>25.392535</td>
          <td>0.307219</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.672845</td>
          <td>1.523275</td>
          <td>29.369086</td>
          <td>1.158310</td>
          <td>27.632094</td>
          <td>0.315650</td>
          <td>26.032283</td>
          <td>0.131645</td>
          <td>25.196512</td>
          <td>0.120417</td>
          <td>24.476007</td>
          <td>0.143006</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.880473</td>
          <td>0.863690</td>
          <td>27.605234</td>
          <td>0.308941</td>
          <td>26.248808</td>
          <td>0.158599</td>
          <td>25.568071</td>
          <td>0.165825</td>
          <td>25.333451</td>
          <td>0.292965</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.843637</td>
          <td>0.489403</td>
          <td>26.025500</td>
          <td>0.091631</td>
          <td>25.987045</td>
          <td>0.077950</td>
          <td>25.636963</td>
          <td>0.093252</td>
          <td>25.770277</td>
          <td>0.196800</td>
          <td>24.828568</td>
          <td>0.193115</td>
          <td>0.010929</td>
          <td>0.009473</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>26.652808</td>
          <td>0.424052</td>
          <td>26.452302</td>
          <td>0.132908</td>
          <td>25.442660</td>
          <td>0.048109</td>
          <td>24.990884</td>
          <td>0.052658</td>
          <td>24.826513</td>
          <td>0.087112</td>
          <td>24.528038</td>
          <td>0.149547</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.672222</td>
          <td>0.866285</td>
          <td>26.459733</td>
          <td>0.133764</td>
          <td>26.030037</td>
          <td>0.080965</td>
          <td>25.110367</td>
          <td>0.058551</td>
          <td>24.764652</td>
          <td>0.082491</td>
          <td>24.176039</td>
          <td>0.110258</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.273075</td>
          <td>0.315332</td>
          <td>26.519538</td>
          <td>0.140842</td>
          <td>26.396426</td>
          <td>0.111670</td>
          <td>26.285186</td>
          <td>0.163605</td>
          <td>26.398575</td>
          <td>0.329244</td>
          <td>26.292717</td>
          <td>0.607000</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.948875</td>
          <td>0.528732</td>
          <td>26.267550</td>
          <td>0.113230</td>
          <td>26.043992</td>
          <td>0.081967</td>
          <td>25.731603</td>
          <td>0.101323</td>
          <td>25.409878</td>
          <td>0.144815</td>
          <td>25.577675</td>
          <td>0.355824</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.027181</td>
          <td>1.075456</td>
          <td>27.062644</td>
          <td>0.223115</td>
          <td>26.393636</td>
          <td>0.111399</td>
          <td>26.499731</td>
          <td>0.196229</td>
          <td>25.861657</td>
          <td>0.212468</td>
          <td>25.694078</td>
          <td>0.389607</td>
          <td>0.059611</td>
          <td>0.049181</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_gaap = errorModel_gaap(samples_truth)
    samples_w_errs_gaap.data



.. parsed-literal::

    Inserting handle into data store.  output_error_model_gaap: inprogress_output_error_model_gaap.pq, error_model_gaap




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398944</td>
          <td>28.581127</td>
          <td>1.554836</td>
          <td>26.469001</td>
          <td>0.155089</td>
          <td>26.099649</td>
          <td>0.101191</td>
          <td>25.153932</td>
          <td>0.072145</td>
          <td>24.580743</td>
          <td>0.082495</td>
          <td>23.927034</td>
          <td>0.104736</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.127518</td>
          <td>0.659830</td>
          <td>28.268833</td>
          <td>0.640591</td>
          <td>26.570732</td>
          <td>0.152274</td>
          <td>26.230051</td>
          <td>0.183766</td>
          <td>25.659243</td>
          <td>0.209201</td>
          <td>25.532521</td>
          <td>0.398291</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.590911</td>
          <td>0.906154</td>
          <td>28.572776</td>
          <td>0.798330</td>
          <td>27.708592</td>
          <td>0.395030</td>
          <td>25.961086</td>
          <td>0.149398</td>
          <td>24.844809</td>
          <td>0.106356</td>
          <td>24.022221</td>
          <td>0.116421</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.257164</td>
          <td>0.315894</td>
          <td>27.223780</td>
          <td>0.280164</td>
          <td>25.949790</td>
          <td>0.154796</td>
          <td>25.584179</td>
          <td>0.209386</td>
          <td>25.392593</td>
          <td>0.379693</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>28.770738</td>
          <td>1.702272</td>
          <td>25.944625</td>
          <td>0.098502</td>
          <td>25.879526</td>
          <td>0.083427</td>
          <td>25.709432</td>
          <td>0.117534</td>
          <td>25.418480</td>
          <td>0.170765</td>
          <td>24.971119</td>
          <td>0.254711</td>
          <td>0.010929</td>
          <td>0.009473</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>26.770623</td>
          <td>0.518597</td>
          <td>26.350218</td>
          <td>0.142686</td>
          <td>25.391084</td>
          <td>0.055294</td>
          <td>25.072879</td>
          <td>0.068642</td>
          <td>24.731536</td>
          <td>0.096192</td>
          <td>24.902662</td>
          <td>0.245608</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.776815</td>
          <td>0.202039</td>
          <td>25.991514</td>
          <td>0.092421</td>
          <td>25.260573</td>
          <td>0.079617</td>
          <td>24.874862</td>
          <td>0.107234</td>
          <td>24.246698</td>
          <td>0.138836</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.448935</td>
          <td>0.405281</td>
          <td>26.960705</td>
          <td>0.237197</td>
          <td>26.323288</td>
          <td>0.124539</td>
          <td>26.181020</td>
          <td>0.178519</td>
          <td>26.095053</td>
          <td>0.302666</td>
          <td>25.281965</td>
          <td>0.331204</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.697458</td>
          <td>0.494747</td>
          <td>26.214447</td>
          <td>0.128101</td>
          <td>26.094237</td>
          <td>0.103917</td>
          <td>25.781875</td>
          <td>0.129201</td>
          <td>25.731667</td>
          <td>0.228875</td>
          <td>24.754076</td>
          <td>0.219374</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.728863</td>
          <td>1.675870</td>
          <td>26.924258</td>
          <td>0.229614</td>
          <td>26.976589</td>
          <td>0.216709</td>
          <td>26.393893</td>
          <td>0.212963</td>
          <td>25.853224</td>
          <td>0.248016</td>
          <td>25.823989</td>
          <td>0.500560</td>
          <td>0.059611</td>
          <td>0.049181</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_auto = errorModel_auto(samples_truth)
    samples_w_errs_auto.data



.. parsed-literal::

    Inserting handle into data store.  output_error_model_auto: inprogress_output_error_model_auto.pq, error_model_auto




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398944</td>
          <td>27.298156</td>
          <td>0.676870</td>
          <td>26.622829</td>
          <td>0.153920</td>
          <td>25.943170</td>
          <td>0.074996</td>
          <td>25.193512</td>
          <td>0.063042</td>
          <td>24.706433</td>
          <td>0.078371</td>
          <td>24.079794</td>
          <td>0.101375</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.592515</td>
          <td>0.405181</td>
          <td>27.983113</td>
          <td>0.463524</td>
          <td>26.699638</td>
          <td>0.145355</td>
          <td>26.068024</td>
          <td>0.135908</td>
          <td>25.700379</td>
          <td>0.185705</td>
          <td>25.628624</td>
          <td>0.370618</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.681736</td>
          <td>0.801110</td>
          <td>28.766362</td>
          <td>0.776417</td>
          <td>26.215568</td>
          <td>0.167517</td>
          <td>25.002261</td>
          <td>0.110265</td>
          <td>24.084901</td>
          <td>0.110776</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.486486</td>
          <td>0.775828</td>
          <td>27.016238</td>
          <td>0.235581</td>
          <td>26.334303</td>
          <td>0.213554</td>
          <td>25.403440</td>
          <td>0.179216</td>
          <td>24.755966</td>
          <td>0.226680</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.700923</td>
          <td>0.440193</td>
          <td>26.132389</td>
          <td>0.100753</td>
          <td>25.930970</td>
          <td>0.074288</td>
          <td>25.680736</td>
          <td>0.097049</td>
          <td>25.667826</td>
          <td>0.180743</td>
          <td>25.656079</td>
          <td>0.378789</td>
          <td>0.010929</td>
          <td>0.009473</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.268256</td>
          <td>0.121310</td>
          <td>25.364935</td>
          <td>0.048631</td>
          <td>25.130440</td>
          <td>0.064780</td>
          <td>24.819396</td>
          <td>0.093663</td>
          <td>24.757030</td>
          <td>0.196607</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.673422</td>
          <td>1.533565</td>
          <td>26.692480</td>
          <td>0.165628</td>
          <td>26.098274</td>
          <td>0.087417</td>
          <td>25.131385</td>
          <td>0.060706</td>
          <td>24.687348</td>
          <td>0.078338</td>
          <td>24.066270</td>
          <td>0.101892</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.012012</td>
          <td>0.569286</td>
          <td>26.790053</td>
          <td>0.184928</td>
          <td>26.222113</td>
          <td>0.100674</td>
          <td>26.091254</td>
          <td>0.145660</td>
          <td>25.777230</td>
          <td>0.207406</td>
          <td>25.494399</td>
          <td>0.348745</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.045816</td>
          <td>0.606067</td>
          <td>26.097677</td>
          <td>0.107943</td>
          <td>26.140823</td>
          <td>0.100112</td>
          <td>25.838806</td>
          <td>0.125292</td>
          <td>25.664322</td>
          <td>0.200874</td>
          <td>25.413238</td>
          <td>0.347679</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.336365</td>
          <td>0.339843</td>
          <td>27.206624</td>
          <td>0.259370</td>
          <td>26.734576</td>
          <td>0.155438</td>
          <td>26.481442</td>
          <td>0.200923</td>
          <td>25.546504</td>
          <td>0.169059</td>
          <td>25.988481</td>
          <td>0.503832</td>
          <td>0.059611</td>
          <td>0.049181</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



Notice some of the magnitudes are inf’s. These are non-detections
(i.e. the noisy flux was negative). You can change the nSigma limit for
non-detections by setting ``sigLim=...``. For example, if ``sigLim=5``,
then all fluxes with ``SNR<5`` are flagged as non-detections.

Let’s plot the error as a function of magnitude

.. code:: ipython3

    %matplotlib inline
    
    fig, axes_ = plt.subplots(ncols=3, nrows=2, figsize=(15, 9), dpi=100)
    axes = axes_.reshape(-1)
    for i, band in enumerate("ugrizy"):
        ax = axes[i]
        # pull out the magnitudes and errors
        mags = samples_w_errs.data[band].to_numpy()
        errs = samples_w_errs.data[band + "_err"].to_numpy()
        
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
        
        # plot errs vs mags
        #ax.plot(mags, errs, label=band) 
        
        #plt.plot(mags, errs, c='C'+str(i))
        ax.scatter(samples_w_errs_gaap.data[band].to_numpy(),
                samples_w_errs_gaap.data[band + "_err"].to_numpy(),
                    s=5, marker='.', color='C0', alpha=0.8, label='GAAP')
        
        ax.plot(mags, errs, color='C3', label='Point source')
        
        
        ax.legend()
        ax.set_xlim(18, 31)
        ax.set_ylim(-0.1, 3.5)
        ax.set(xlabel=band+" Band Magnitude (AB)", ylabel="Error (mags)")




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_24_0.png


.. code:: ipython3

    %matplotlib inline
    
    fig, axes_ = plt.subplots(ncols=3, nrows=2, figsize=(15, 9), dpi=100)
    axes = axes_.reshape(-1)
    for i, band in enumerate("ugrizy"):
        ax = axes[i]
        # pull out the magnitudes and errors
        mags = samples_w_errs.data[band].to_numpy()
        errs = samples_w_errs.data[band + "_err"].to_numpy()
        
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
        
        # plot errs vs mags
        #ax.plot(mags, errs, label=band) 
        
        #plt.plot(mags, errs, c='C'+str(i))
        ax.scatter(samples_w_errs_auto.data[band].to_numpy(),
                samples_w_errs_auto.data[band + "_err"].to_numpy(),
                    s=5, marker='.', color='C0', alpha=0.8, label='AUTO')
        
        ax.plot(mags, errs, color='C3', label='Point source')
        
        
        ax.legend()
        ax.set_xlim(18, 31)
        ax.set_ylim(-0.1, 3.5)
        ax.set(xlabel=band+" Band Magnitude (AB)", ylabel="Error (mags)")




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_25_0.png


You can see that the photometric error increases as magnitude gets
dimmer, just like you would expect, and that the extended source errors
are greater than the point source errors. The extended source errors are
also scattered, because the galaxies have random sizes.

Also, you can find the GAaP and AUTO magnitude error are scattered due
to variable galaxy sizes. Also, you can find that there are gaps between
GAAP magnitude error and point souce magnitude error, this is because
the additional factors due to aperture sizes have a minimum value of
:math:`\sqrt{(\sigma^2+A_{\mathrm{min}})/\sigma^2}`, where
:math:`\sigma` is the width of the beam, :math:`A_{\min}` is an offset
of the aperture sizes (taken to be 0.7 arcmin here).

You can also see that there are *very* faint galaxies in this sample.
That’s because, by default, the error model returns magnitudes for all
positive fluxes. If you want these galaxies flagged as non-detections
instead, you can set e.g. ``sigLim=5``, and everything with ``SNR<5``
will be flagged as a non-detection.
