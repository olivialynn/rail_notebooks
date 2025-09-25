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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f3d26be9720>



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
    0      23.994413  0.033848  0.018175  
    1      25.391064  0.136471  0.115711  
    2      24.304707  0.236806  0.191228  
    3      25.291103  0.097813  0.056587  
    4      25.096743  0.111091  0.088591  
    ...          ...       ...       ...  
    99995  24.737946  0.120902  0.086340  
    99996  24.224169  0.009799  0.008682  
    99997  25.613836  0.163452  0.122135  
    99998  25.274899  0.128304  0.093878  
    99999  25.699642  0.117608  0.100400  
    
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

    Inserting handle into data store.  input: None, error_model
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
          <td>27.990850</td>
          <td>1.052769</td>
          <td>26.756332</td>
          <td>0.172464</td>
          <td>26.021024</td>
          <td>0.080323</td>
          <td>25.185166</td>
          <td>0.062568</td>
          <td>24.771115</td>
          <td>0.082962</td>
          <td>23.901414</td>
          <td>0.086665</td>
          <td>0.033848</td>
          <td>0.018175</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.196909</td>
          <td>0.631065</td>
          <td>27.789315</td>
          <td>0.399785</td>
          <td>26.652422</td>
          <td>0.139434</td>
          <td>26.357202</td>
          <td>0.173951</td>
          <td>26.222076</td>
          <td>0.285807</td>
          <td>25.443464</td>
          <td>0.319984</td>
          <td>0.136471</td>
          <td>0.115711</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.850180</td>
          <td>0.491779</td>
          <td>27.832523</td>
          <td>0.413268</td>
          <td>28.683011</td>
          <td>0.689897</td>
          <td>25.893059</td>
          <td>0.116663</td>
          <td>24.846726</td>
          <td>0.088675</td>
          <td>24.183125</td>
          <td>0.110941</td>
          <td>0.236806</td>
          <td>0.191228</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.881575</td>
          <td>0.169684</td>
          <td>26.159370</td>
          <td>0.146892</td>
          <td>25.478238</td>
          <td>0.153569</td>
          <td>25.600519</td>
          <td>0.362252</td>
          <td>0.097813</td>
          <td>0.056587</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.759643</td>
          <td>0.459707</td>
          <td>25.920306</td>
          <td>0.083539</td>
          <td>25.933666</td>
          <td>0.074359</td>
          <td>25.735473</td>
          <td>0.101667</td>
          <td>25.838793</td>
          <td>0.208445</td>
          <td>25.595178</td>
          <td>0.360741</td>
          <td>0.111091</td>
          <td>0.088591</td>
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
          <td>26.748752</td>
          <td>0.455965</td>
          <td>26.395429</td>
          <td>0.126528</td>
          <td>25.367782</td>
          <td>0.045015</td>
          <td>24.994543</td>
          <td>0.052830</td>
          <td>24.819277</td>
          <td>0.086558</td>
          <td>25.416621</td>
          <td>0.313200</td>
          <td>0.120902</td>
          <td>0.086340</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.472657</td>
          <td>0.135264</td>
          <td>26.038051</td>
          <td>0.081539</td>
          <td>25.180519</td>
          <td>0.062311</td>
          <td>24.704517</td>
          <td>0.078228</td>
          <td>24.209526</td>
          <td>0.113525</td>
          <td>0.009799</td>
          <td>0.008682</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.989177</td>
          <td>1.051731</td>
          <td>26.575562</td>
          <td>0.147791</td>
          <td>26.499860</td>
          <td>0.122189</td>
          <td>26.018232</td>
          <td>0.130054</td>
          <td>26.021444</td>
          <td>0.242600</td>
          <td>25.592456</td>
          <td>0.359972</td>
          <td>0.163452</td>
          <td>0.122135</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.257146</td>
          <td>0.311347</td>
          <td>26.158517</td>
          <td>0.102955</td>
          <td>26.204237</td>
          <td>0.094382</td>
          <td>25.817501</td>
          <td>0.109227</td>
          <td>25.547813</td>
          <td>0.162985</td>
          <td>25.485303</td>
          <td>0.330810</td>
          <td>0.128304</td>
          <td>0.093878</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.491863</td>
          <td>0.770918</td>
          <td>26.750934</td>
          <td>0.171675</td>
          <td>26.297417</td>
          <td>0.102416</td>
          <td>26.477302</td>
          <td>0.192558</td>
          <td>25.845487</td>
          <td>0.209615</td>
          <td>25.615629</td>
          <td>0.366557</td>
          <td>0.117608</td>
          <td>0.100400</td>
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
          <td>31.376336</td>
          <td>4.081279</td>
          <td>26.482386</td>
          <td>0.157216</td>
          <td>25.863834</td>
          <td>0.082459</td>
          <td>25.242431</td>
          <td>0.078214</td>
          <td>24.757505</td>
          <td>0.096604</td>
          <td>23.999895</td>
          <td>0.111896</td>
          <td>0.033848</td>
          <td>0.018175</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.404687</td>
          <td>0.818293</td>
          <td>27.796387</td>
          <td>0.473338</td>
          <td>26.550173</td>
          <td>0.157300</td>
          <td>26.523958</td>
          <td>0.247010</td>
          <td>25.956113</td>
          <td>0.280364</td>
          <td>25.832159</td>
          <td>0.521816</td>
          <td>0.136471</td>
          <td>0.115711</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.519618</td>
          <td>0.923466</td>
          <td>29.202783</td>
          <td>1.250154</td>
          <td>29.232358</td>
          <td>1.196432</td>
          <td>25.747873</td>
          <td>0.139444</td>
          <td>24.947290</td>
          <td>0.130086</td>
          <td>24.194245</td>
          <td>0.151429</td>
          <td>0.236806</td>
          <td>0.191228</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.536339</td>
          <td>1.534783</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.125485</td>
          <td>0.247729</td>
          <td>26.508990</td>
          <td>0.236890</td>
          <td>25.722035</td>
          <td>0.224862</td>
          <td>25.753896</td>
          <td>0.479817</td>
          <td>0.097813</td>
          <td>0.056587</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.476365</td>
          <td>0.419549</td>
          <td>26.274615</td>
          <td>0.135144</td>
          <td>25.826257</td>
          <td>0.082270</td>
          <td>25.442863</td>
          <td>0.096316</td>
          <td>25.352037</td>
          <td>0.166586</td>
          <td>25.418258</td>
          <td>0.375591</td>
          <td>0.111091</td>
          <td>0.088591</td>
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
          <td>26.478339</td>
          <td>0.421004</td>
          <td>26.637066</td>
          <td>0.184656</td>
          <td>25.381915</td>
          <td>0.055689</td>
          <td>24.856394</td>
          <td>0.057559</td>
          <td>25.200419</td>
          <td>0.146732</td>
          <td>24.660923</td>
          <td>0.203866</td>
          <td>0.120902</td>
          <td>0.086340</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.544071</td>
          <td>0.432196</td>
          <td>26.676347</td>
          <td>0.185031</td>
          <td>26.016637</td>
          <td>0.094113</td>
          <td>25.277549</td>
          <td>0.080489</td>
          <td>25.022594</td>
          <td>0.121485</td>
          <td>24.605852</td>
          <td>0.187907</td>
          <td>0.009799</td>
          <td>0.008682</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.390677</td>
          <td>0.817597</td>
          <td>26.607605</td>
          <td>0.184877</td>
          <td>26.291325</td>
          <td>0.127646</td>
          <td>26.205997</td>
          <td>0.192187</td>
          <td>26.095393</td>
          <td>0.317713</td>
          <td>24.934267</td>
          <td>0.263100</td>
          <td>0.163452</td>
          <td>0.122135</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.332844</td>
          <td>0.776146</td>
          <td>26.075365</td>
          <td>0.114532</td>
          <td>26.052581</td>
          <td>0.101167</td>
          <td>25.892712</td>
          <td>0.143570</td>
          <td>25.885446</td>
          <td>0.262128</td>
          <td>25.202691</td>
          <td>0.319243</td>
          <td>0.128304</td>
          <td>0.093878</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.207246</td>
          <td>0.712994</td>
          <td>26.689766</td>
          <td>0.193550</td>
          <td>26.543876</td>
          <td>0.154556</td>
          <td>25.954534</td>
          <td>0.151093</td>
          <td>25.973486</td>
          <td>0.281077</td>
          <td>26.361413</td>
          <td>0.747729</td>
          <td>0.117608</td>
          <td>0.100400</td>
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
          <td>26.831215</td>
          <td>0.487739</td>
          <td>26.553106</td>
          <td>0.146193</td>
          <td>26.030810</td>
          <td>0.081830</td>
          <td>25.208805</td>
          <td>0.064568</td>
          <td>24.754623</td>
          <td>0.082581</td>
          <td>23.850006</td>
          <td>0.083688</td>
          <td>0.033848</td>
          <td>0.018175</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.039483</td>
          <td>0.551260</td>
          <td>26.399320</td>
          <td>0.133584</td>
          <td>26.481271</td>
          <td>0.230613</td>
          <td>26.002414</td>
          <td>0.281994</td>
          <td>25.524295</td>
          <td>0.401845</td>
          <td>0.136471</td>
          <td>0.115711</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.966985</td>
          <td>1.265878</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.388103</td>
          <td>0.766977</td>
          <td>26.092390</td>
          <td>0.206542</td>
          <td>25.336318</td>
          <td>0.199948</td>
          <td>24.251538</td>
          <td>0.175698</td>
          <td>0.236806</td>
          <td>0.191228</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.315516</td>
          <td>0.715050</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.439077</td>
          <td>0.290381</td>
          <td>26.581798</td>
          <td>0.227188</td>
          <td>25.402900</td>
          <td>0.155389</td>
          <td>24.970270</td>
          <td>0.234796</td>
          <td>0.097813</td>
          <td>0.056587</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.282823</td>
          <td>0.343521</td>
          <td>26.165559</td>
          <td>0.115072</td>
          <td>26.000713</td>
          <td>0.089005</td>
          <td>25.632757</td>
          <td>0.105294</td>
          <td>25.510171</td>
          <td>0.177286</td>
          <td>25.422422</td>
          <td>0.351941</td>
          <td>0.111091</td>
          <td>0.088591</td>
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
          <td>27.106033</td>
          <td>0.637723</td>
          <td>26.261726</td>
          <td>0.126096</td>
          <td>25.429211</td>
          <td>0.054179</td>
          <td>25.102917</td>
          <td>0.066650</td>
          <td>24.803463</td>
          <td>0.097124</td>
          <td>25.061245</td>
          <td>0.265699</td>
          <td>0.120902</td>
          <td>0.086340</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.798273</td>
          <td>0.178885</td>
          <td>26.115736</td>
          <td>0.087418</td>
          <td>25.123555</td>
          <td>0.059313</td>
          <td>24.826429</td>
          <td>0.087207</td>
          <td>24.296615</td>
          <td>0.122605</td>
          <td>0.009799</td>
          <td>0.008682</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.128400</td>
          <td>0.325622</td>
          <td>26.640816</td>
          <td>0.188873</td>
          <td>26.252984</td>
          <td>0.122527</td>
          <td>26.259965</td>
          <td>0.199560</td>
          <td>25.927779</td>
          <td>0.275627</td>
          <td>25.660313</td>
          <td>0.462012</td>
          <td>0.163452</td>
          <td>0.122135</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>28.042995</td>
          <td>1.162562</td>
          <td>26.470373</td>
          <td>0.153085</td>
          <td>26.146902</td>
          <td>0.103761</td>
          <td>25.821479</td>
          <td>0.127344</td>
          <td>25.695696</td>
          <td>0.212313</td>
          <td>24.701953</td>
          <td>0.200432</td>
          <td>0.128304</td>
          <td>0.093878</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.446476</td>
          <td>0.395309</td>
          <td>26.433899</td>
          <td>0.147617</td>
          <td>26.705454</td>
          <td>0.167199</td>
          <td>26.305831</td>
          <td>0.191616</td>
          <td>25.962445</td>
          <td>0.263259</td>
          <td>25.878099</td>
          <td>0.506718</td>
          <td>0.117608</td>
          <td>0.100400</td>
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
