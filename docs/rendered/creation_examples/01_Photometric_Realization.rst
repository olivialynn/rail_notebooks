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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.17/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f7234890d60>



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
          <td>28.114513</td>
          <td>1.131148</td>
          <td>26.763627</td>
          <td>0.173536</td>
          <td>25.943127</td>
          <td>0.074983</td>
          <td>25.141358</td>
          <td>0.060183</td>
          <td>24.718420</td>
          <td>0.079194</td>
          <td>23.995736</td>
          <td>0.094159</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.547609</td>
          <td>0.799597</td>
          <td>27.952885</td>
          <td>0.452804</td>
          <td>26.577040</td>
          <td>0.130645</td>
          <td>26.580529</td>
          <td>0.209991</td>
          <td>25.803941</td>
          <td>0.202445</td>
          <td>25.156542</td>
          <td>0.253691</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.954042</td>
          <td>0.123015</td>
          <td>25.045514</td>
          <td>0.105566</td>
          <td>24.549521</td>
          <td>0.152329</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>33.314872</td>
          <td>4.660968</td>
          <td>27.068299</td>
          <td>0.198717</td>
          <td>26.228700</td>
          <td>0.155893</td>
          <td>25.451563</td>
          <td>0.150096</td>
          <td>24.809759</td>
          <td>0.190077</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.382070</td>
          <td>0.343794</td>
          <td>26.152062</td>
          <td>0.102376</td>
          <td>25.994398</td>
          <td>0.078458</td>
          <td>25.811468</td>
          <td>0.108653</td>
          <td>25.457360</td>
          <td>0.150844</td>
          <td>24.685488</td>
          <td>0.171085</td>
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
          <td>26.483878</td>
          <td>0.372331</td>
          <td>26.317850</td>
          <td>0.118293</td>
          <td>25.441880</td>
          <td>0.048076</td>
          <td>25.056575</td>
          <td>0.055821</td>
          <td>24.714938</td>
          <td>0.078951</td>
          <td>24.752938</td>
          <td>0.181164</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>29.371311</td>
          <td>2.086981</td>
          <td>26.710874</td>
          <td>0.165922</td>
          <td>26.050197</td>
          <td>0.082417</td>
          <td>25.163790</td>
          <td>0.061393</td>
          <td>24.917588</td>
          <td>0.094373</td>
          <td>24.129733</td>
          <td>0.105888</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.124230</td>
          <td>0.599653</td>
          <td>26.798371</td>
          <td>0.178726</td>
          <td>26.423243</td>
          <td>0.114312</td>
          <td>26.333206</td>
          <td>0.170438</td>
          <td>25.722088</td>
          <td>0.188970</td>
          <td>27.595878</td>
          <td>1.356141</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.520795</td>
          <td>0.383159</td>
          <td>26.232178</td>
          <td>0.109793</td>
          <td>26.110812</td>
          <td>0.086939</td>
          <td>25.879167</td>
          <td>0.115261</td>
          <td>25.415007</td>
          <td>0.145456</td>
          <td>25.192005</td>
          <td>0.261168</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.178664</td>
          <td>0.623067</td>
          <td>26.608873</td>
          <td>0.152074</td>
          <td>26.458345</td>
          <td>0.117859</td>
          <td>26.252103</td>
          <td>0.159046</td>
          <td>25.854468</td>
          <td>0.211195</td>
          <td>26.388196</td>
          <td>0.648908</td>
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
          <td>27.263379</td>
          <td>0.723676</td>
          <td>26.814848</td>
          <td>0.207831</td>
          <td>25.880875</td>
          <td>0.083499</td>
          <td>25.139301</td>
          <td>0.071218</td>
          <td>24.671081</td>
          <td>0.089323</td>
          <td>23.967449</td>
          <td>0.108499</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.425928</td>
          <td>0.394845</td>
          <td>27.917107</td>
          <td>0.497761</td>
          <td>26.673694</td>
          <td>0.166282</td>
          <td>26.343007</td>
          <td>0.202113</td>
          <td>25.787886</td>
          <td>0.232841</td>
          <td>26.009415</td>
          <td>0.568064</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.183409</td>
          <td>0.694652</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.085849</td>
          <td>0.524494</td>
          <td>26.247897</td>
          <td>0.190700</td>
          <td>24.969616</td>
          <td>0.118577</td>
          <td>24.021330</td>
          <td>0.116331</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.380337</td>
          <td>0.812734</td>
          <td>27.760843</td>
          <td>0.466527</td>
          <td>27.339037</td>
          <td>0.307445</td>
          <td>26.115853</td>
          <td>0.178325</td>
          <td>25.100619</td>
          <td>0.138818</td>
          <td>25.927906</td>
          <td>0.566676</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.996983</td>
          <td>0.281291</td>
          <td>26.227284</td>
          <td>0.125989</td>
          <td>25.801597</td>
          <td>0.077886</td>
          <td>25.590056</td>
          <td>0.105916</td>
          <td>25.421486</td>
          <td>0.171202</td>
          <td>24.855320</td>
          <td>0.231525</td>
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
          <td>30.568845</td>
          <td>3.316697</td>
          <td>26.523445</td>
          <td>0.165491</td>
          <td>25.527213</td>
          <td>0.062388</td>
          <td>25.233639</td>
          <td>0.079120</td>
          <td>24.868361</td>
          <td>0.108425</td>
          <td>24.476093</td>
          <td>0.171840</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.367235</td>
          <td>0.777245</td>
          <td>26.639093</td>
          <td>0.179903</td>
          <td>26.069608</td>
          <td>0.098975</td>
          <td>25.216513</td>
          <td>0.076580</td>
          <td>24.935066</td>
          <td>0.113016</td>
          <td>24.209969</td>
          <td>0.134505</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.002819</td>
          <td>0.609527</td>
          <td>26.663411</td>
          <td>0.185022</td>
          <td>26.387233</td>
          <td>0.131631</td>
          <td>26.316606</td>
          <td>0.200157</td>
          <td>25.839492</td>
          <td>0.245861</td>
          <td>25.389422</td>
          <td>0.360478</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.441522</td>
          <td>0.828421</td>
          <td>26.308128</td>
          <td>0.138890</td>
          <td>26.048654</td>
          <td>0.099853</td>
          <td>25.999797</td>
          <td>0.155865</td>
          <td>25.520108</td>
          <td>0.191766</td>
          <td>25.198958</td>
          <td>0.315446</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.253114</td>
          <td>1.320795</td>
          <td>26.797233</td>
          <td>0.206565</td>
          <td>26.487689</td>
          <td>0.143160</td>
          <td>26.509873</td>
          <td>0.234516</td>
          <td>25.992817</td>
          <td>0.277983</td>
          <td>26.931561</td>
          <td>1.047412</td>
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
          <td>27.462167</td>
          <td>0.755984</td>
          <td>26.530890</td>
          <td>0.142240</td>
          <td>26.073810</td>
          <td>0.084162</td>
          <td>25.216165</td>
          <td>0.064321</td>
          <td>24.608183</td>
          <td>0.071853</td>
          <td>24.075606</td>
          <td>0.101004</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.692675</td>
          <td>0.437331</td>
          <td>28.315677</td>
          <td>0.590880</td>
          <td>26.699455</td>
          <td>0.145332</td>
          <td>26.271586</td>
          <td>0.161873</td>
          <td>26.673912</td>
          <td>0.408559</td>
          <td>26.069995</td>
          <td>0.517599</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.184058</td>
          <td>0.655054</td>
          <td>29.324373</td>
          <td>1.182760</td>
          <td>27.788573</td>
          <td>0.384172</td>
          <td>26.237630</td>
          <td>0.170693</td>
          <td>25.265607</td>
          <td>0.138569</td>
          <td>24.267098</td>
          <td>0.129779</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.048808</td>
          <td>0.574396</td>
          <td>27.168116</td>
          <td>0.266880</td>
          <td>26.058652</td>
          <td>0.169265</td>
          <td>25.528037</td>
          <td>0.199082</td>
          <td>25.861988</td>
          <td>0.538735</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.055148</td>
          <td>0.264738</td>
          <td>26.108123</td>
          <td>0.098636</td>
          <td>25.893168</td>
          <td>0.071845</td>
          <td>25.711606</td>
          <td>0.099712</td>
          <td>25.807696</td>
          <td>0.203359</td>
          <td>25.138226</td>
          <td>0.250249</td>
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
          <td>26.716802</td>
          <td>0.466893</td>
          <td>26.681649</td>
          <td>0.173028</td>
          <td>25.470316</td>
          <td>0.053399</td>
          <td>25.187639</td>
          <td>0.068147</td>
          <td>24.855192</td>
          <td>0.096652</td>
          <td>24.695172</td>
          <td>0.186618</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.186291</td>
          <td>0.632140</td>
          <td>26.464462</td>
          <td>0.136215</td>
          <td>26.027503</td>
          <td>0.082133</td>
          <td>25.331967</td>
          <td>0.072511</td>
          <td>24.911311</td>
          <td>0.095412</td>
          <td>24.080345</td>
          <td>0.103155</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.014295</td>
          <td>0.570218</td>
          <td>26.516512</td>
          <td>0.146478</td>
          <td>26.536565</td>
          <td>0.132375</td>
          <td>26.282784</td>
          <td>0.171585</td>
          <td>26.130105</td>
          <td>0.277500</td>
          <td>25.360029</td>
          <td>0.313481</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.947031</td>
          <td>0.261389</td>
          <td>26.463675</td>
          <td>0.148175</td>
          <td>26.116536</td>
          <td>0.098003</td>
          <td>26.007154</td>
          <td>0.144900</td>
          <td>25.618516</td>
          <td>0.193284</td>
          <td>25.206805</td>
          <td>0.294945</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.382998</td>
          <td>0.732062</td>
          <td>26.833732</td>
          <td>0.190255</td>
          <td>26.670198</td>
          <td>0.147087</td>
          <td>26.122608</td>
          <td>0.148117</td>
          <td>26.042183</td>
          <td>0.255907</td>
          <td>25.887718</td>
          <td>0.467520</td>
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




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_24_0.png


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




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_25_0.png


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
