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

    <pzflow.flow.Flow at 0x7f82a3124a60>



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
          <td>28.018702</td>
          <td>1.070136</td>
          <td>26.834794</td>
          <td>0.184320</td>
          <td>25.996170</td>
          <td>0.078580</td>
          <td>25.203310</td>
          <td>0.063583</td>
          <td>24.718951</td>
          <td>0.079232</td>
          <td>23.836286</td>
          <td>0.081832</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.351266</td>
          <td>0.335535</td>
          <td>26.724025</td>
          <td>0.167791</td>
          <td>26.417206</td>
          <td>0.113712</td>
          <td>26.202666</td>
          <td>0.152454</td>
          <td>25.800484</td>
          <td>0.201859</td>
          <td>25.276286</td>
          <td>0.279727</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.286089</td>
          <td>1.245197</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.832944</td>
          <td>0.369891</td>
          <td>26.007340</td>
          <td>0.128833</td>
          <td>25.088788</td>
          <td>0.109633</td>
          <td>24.571321</td>
          <td>0.155201</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.916036</td>
          <td>1.546273</td>
          <td>27.712259</td>
          <td>0.336424</td>
          <td>25.991895</td>
          <td>0.127121</td>
          <td>25.457434</td>
          <td>0.150854</td>
          <td>25.860249</td>
          <td>0.442411</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.528840</td>
          <td>0.385554</td>
          <td>25.983244</td>
          <td>0.088293</td>
          <td>25.947028</td>
          <td>0.075242</td>
          <td>25.767849</td>
          <td>0.104589</td>
          <td>25.713387</td>
          <td>0.187587</td>
          <td>25.171729</td>
          <td>0.256869</td>
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
          <td>26.324766</td>
          <td>0.328567</td>
          <td>26.292955</td>
          <td>0.115761</td>
          <td>25.515524</td>
          <td>0.051325</td>
          <td>25.076421</td>
          <td>0.056813</td>
          <td>24.897208</td>
          <td>0.092699</td>
          <td>24.978466</td>
          <td>0.218956</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.768060</td>
          <td>0.919984</td>
          <td>27.138385</td>
          <td>0.237565</td>
          <td>26.116595</td>
          <td>0.087382</td>
          <td>25.191446</td>
          <td>0.062917</td>
          <td>24.851132</td>
          <td>0.089019</td>
          <td>24.185521</td>
          <td>0.111173</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.695083</td>
          <td>0.878905</td>
          <td>26.588330</td>
          <td>0.149420</td>
          <td>26.370147</td>
          <td>0.109139</td>
          <td>25.900825</td>
          <td>0.117454</td>
          <td>26.252931</td>
          <td>0.293020</td>
          <td>26.788611</td>
          <td>0.847405</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.587294</td>
          <td>0.403329</td>
          <td>26.355005</td>
          <td>0.122172</td>
          <td>26.088936</td>
          <td>0.085280</td>
          <td>26.124850</td>
          <td>0.142594</td>
          <td>25.571840</td>
          <td>0.166359</td>
          <td>25.300260</td>
          <td>0.285214</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.578995</td>
          <td>0.400765</td>
          <td>26.941419</td>
          <td>0.201635</td>
          <td>26.590241</td>
          <td>0.132145</td>
          <td>26.240124</td>
          <td>0.157425</td>
          <td>26.021538</td>
          <td>0.242619</td>
          <td>25.094570</td>
          <td>0.241080</td>
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
          <td>27.505736</td>
          <td>0.848249</td>
          <td>26.815716</td>
          <td>0.207983</td>
          <td>25.878396</td>
          <td>0.083317</td>
          <td>25.107892</td>
          <td>0.069266</td>
          <td>24.512489</td>
          <td>0.077675</td>
          <td>24.145297</td>
          <td>0.126650</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.420371</td>
          <td>1.434849</td>
          <td>27.989677</td>
          <td>0.524997</td>
          <td>26.556716</td>
          <td>0.150454</td>
          <td>26.438314</td>
          <td>0.218876</td>
          <td>25.856600</td>
          <td>0.246429</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.012754</td>
          <td>1.164436</td>
          <td>28.506392</td>
          <td>0.764287</td>
          <td>27.967407</td>
          <td>0.480649</td>
          <td>25.855981</td>
          <td>0.136476</td>
          <td>25.132916</td>
          <td>0.136594</td>
          <td>24.619669</td>
          <td>0.194283</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>30.734137</td>
          <td>3.513772</td>
          <td>27.397784</td>
          <td>0.353101</td>
          <td>27.564199</td>
          <td>0.367406</td>
          <td>26.245625</td>
          <td>0.198966</td>
          <td>25.405023</td>
          <td>0.180075</td>
          <td>27.158029</td>
          <td>1.237867</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.000149</td>
          <td>0.282013</td>
          <td>26.165556</td>
          <td>0.119424</td>
          <td>26.062517</td>
          <td>0.097985</td>
          <td>25.933403</td>
          <td>0.142675</td>
          <td>25.559887</td>
          <td>0.192482</td>
          <td>24.817314</td>
          <td>0.224338</td>
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
          <td>26.634128</td>
          <td>0.468827</td>
          <td>26.491735</td>
          <td>0.161077</td>
          <td>25.431963</td>
          <td>0.057336</td>
          <td>25.076420</td>
          <td>0.068857</td>
          <td>24.688230</td>
          <td>0.092605</td>
          <td>24.485564</td>
          <td>0.173229</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.282262</td>
          <td>0.734666</td>
          <td>26.324452</td>
          <td>0.137490</td>
          <td>26.105393</td>
          <td>0.102125</td>
          <td>25.247978</td>
          <td>0.078737</td>
          <td>24.851067</td>
          <td>0.105028</td>
          <td>24.265460</td>
          <td>0.141099</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.964172</td>
          <td>0.593110</td>
          <td>26.571532</td>
          <td>0.171166</td>
          <td>26.403445</td>
          <td>0.133488</td>
          <td>26.351819</td>
          <td>0.206156</td>
          <td>26.257805</td>
          <td>0.344525</td>
          <td>25.691599</td>
          <td>0.454638</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.831900</td>
          <td>0.251392</td>
          <td>26.270964</td>
          <td>0.134511</td>
          <td>26.337064</td>
          <td>0.128374</td>
          <td>26.122933</td>
          <td>0.173124</td>
          <td>25.555297</td>
          <td>0.197531</td>
          <td>25.334570</td>
          <td>0.351237</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.452458</td>
          <td>1.464755</td>
          <td>26.707354</td>
          <td>0.191550</td>
          <td>26.258496</td>
          <td>0.117404</td>
          <td>26.500413</td>
          <td>0.232687</td>
          <td>25.686248</td>
          <td>0.215981</td>
          <td>25.484880</td>
          <td>0.387325</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.679594</td>
          <td>0.161574</td>
          <td>26.073124</td>
          <td>0.084111</td>
          <td>25.234940</td>
          <td>0.065400</td>
          <td>24.596991</td>
          <td>0.071145</td>
          <td>23.953278</td>
          <td>0.090724</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.339612</td>
          <td>0.696587</td>
          <td>27.152396</td>
          <td>0.240514</td>
          <td>26.547948</td>
          <td>0.127513</td>
          <td>26.155619</td>
          <td>0.146562</td>
          <td>25.866029</td>
          <td>0.213437</td>
          <td>25.359879</td>
          <td>0.299538</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.378790</td>
          <td>0.653621</td>
          <td>28.037302</td>
          <td>0.464387</td>
          <td>25.795212</td>
          <td>0.116617</td>
          <td>24.941465</td>
          <td>0.104563</td>
          <td>24.629541</td>
          <td>0.177072</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.010954</td>
          <td>0.559014</td>
          <td>27.131854</td>
          <td>0.259089</td>
          <td>26.286269</td>
          <td>0.205145</td>
          <td>25.643490</td>
          <td>0.219266</td>
          <td>25.786854</td>
          <td>0.509975</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.352785</td>
          <td>0.336236</td>
          <td>26.230597</td>
          <td>0.109776</td>
          <td>26.043453</td>
          <td>0.082046</td>
          <td>25.632651</td>
          <td>0.093038</td>
          <td>25.422713</td>
          <td>0.146626</td>
          <td>24.826496</td>
          <td>0.193051</td>
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
          <td>26.609159</td>
          <td>0.430522</td>
          <td>26.417137</td>
          <td>0.137980</td>
          <td>25.346375</td>
          <td>0.047836</td>
          <td>25.032227</td>
          <td>0.059378</td>
          <td>24.871963</td>
          <td>0.098084</td>
          <td>24.664022</td>
          <td>0.181767</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.717852</td>
          <td>0.898919</td>
          <td>26.597050</td>
          <td>0.152660</td>
          <td>25.952180</td>
          <td>0.076850</td>
          <td>25.266779</td>
          <td>0.068446</td>
          <td>25.040342</td>
          <td>0.106826</td>
          <td>24.153449</td>
          <td>0.109960</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.542303</td>
          <td>0.401541</td>
          <td>27.205642</td>
          <td>0.261295</td>
          <td>26.316998</td>
          <td>0.109383</td>
          <td>26.629440</td>
          <td>0.229601</td>
          <td>26.430140</td>
          <td>0.352706</td>
          <td>25.270955</td>
          <td>0.291838</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.586562</td>
          <td>0.432935</td>
          <td>26.323787</td>
          <td>0.131359</td>
          <td>26.100325</td>
          <td>0.096620</td>
          <td>25.731189</td>
          <td>0.114104</td>
          <td>25.796832</td>
          <td>0.224385</td>
          <td>25.198048</td>
          <td>0.292870</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.546778</td>
          <td>0.400401</td>
          <td>26.530066</td>
          <td>0.146922</td>
          <td>26.486559</td>
          <td>0.125521</td>
          <td>26.346715</td>
          <td>0.179336</td>
          <td>26.123513</td>
          <td>0.273479</td>
          <td>25.360695</td>
          <td>0.310708</td>
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
