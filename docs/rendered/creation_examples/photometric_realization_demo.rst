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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.15/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f8c9438f460>



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
    0      0.890625  27.370831  26.712660  26.025223  25.327185  25.016500   
    1      1.978239  29.557047  28.361183  27.587227  27.238544  26.628105   
    2      0.974287  26.566013  25.937716  24.787411  23.872454  23.139563   
    3      1.317978  29.042736  28.274597  27.501110  26.648792  26.091452   
    4      1.386366  26.292624  25.774778  25.429960  24.806530  24.367950   
    ...         ...        ...        ...        ...        ...        ...   
    99995  2.147172  26.550978  26.349937  26.135286  26.082020  25.911032   
    99996  1.457508  27.362209  27.036276  26.823141  26.420132  26.110037   
    99997  1.372993  27.736042  27.271955  26.887583  26.416138  26.043432   
    99998  0.855022  28.044554  27.327116  26.599014  25.862329  25.592169   
    99999  1.723768  27.049067  26.526747  26.094597  25.642973  25.197958   
    
                   y     major     minor  
    0      24.926819  0.003319  0.002869  
    1      26.248560  0.008733  0.007945  
    2      22.832047  0.103938  0.052162  
    3      25.346504  0.147522  0.143359  
    4      23.700008  0.010929  0.009473  
    ...          ...       ...       ...  
    99995  25.558136  0.086491  0.071701  
    99996  25.524906  0.044537  0.022302  
    99997  25.456163  0.073146  0.047825  
    99998  25.506388  0.100551  0.094662  
    99999  24.900501  0.059611  0.049181  
    
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
          <td>0.890625</td>
          <td>28.268291</td>
          <td>1.233087</td>
          <td>26.837510</td>
          <td>0.184744</td>
          <td>26.077978</td>
          <td>0.084460</td>
          <td>25.405388</td>
          <td>0.076038</td>
          <td>24.941783</td>
          <td>0.096399</td>
          <td>24.811817</td>
          <td>0.190407</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.154263</td>
          <td>1.157030</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.238281</td>
          <td>0.229023</td>
          <td>27.234906</td>
          <td>0.357253</td>
          <td>27.476606</td>
          <td>0.728272</td>
          <td>25.659748</td>
          <td>0.379375</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.632986</td>
          <td>0.417692</td>
          <td>25.955431</td>
          <td>0.086160</td>
          <td>24.770106</td>
          <td>0.026576</td>
          <td>23.879598</td>
          <td>0.019884</td>
          <td>23.146350</td>
          <td>0.019889</td>
          <td>22.860224</td>
          <td>0.034466</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.125176</td>
          <td>0.514638</td>
          <td>27.245535</td>
          <td>0.230404</td>
          <td>26.724550</td>
          <td>0.236707</td>
          <td>26.588205</td>
          <td>0.382098</td>
          <td>25.548668</td>
          <td>0.347802</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>27.952238</td>
          <td>1.028971</td>
          <td>25.737669</td>
          <td>0.071111</td>
          <td>25.447625</td>
          <td>0.048322</td>
          <td>24.818951</td>
          <td>0.045203</td>
          <td>24.388840</td>
          <td>0.059154</td>
          <td>23.827212</td>
          <td>0.081180</td>
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
          <td>2.147172</td>
          <td>28.645097</td>
          <td>1.502419</td>
          <td>26.416227</td>
          <td>0.128827</td>
          <td>26.204202</td>
          <td>0.094379</td>
          <td>26.299261</td>
          <td>0.165581</td>
          <td>25.886606</td>
          <td>0.216938</td>
          <td>25.924315</td>
          <td>0.464265</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.651600</td>
          <td>0.423662</td>
          <td>27.329027</td>
          <td>0.277705</td>
          <td>26.985387</td>
          <td>0.185303</td>
          <td>26.303698</td>
          <td>0.166208</td>
          <td>25.916814</td>
          <td>0.222464</td>
          <td>24.882040</td>
          <td>0.201996</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.078617</td>
          <td>0.580548</td>
          <td>27.268732</td>
          <td>0.264405</td>
          <td>26.904072</td>
          <td>0.172962</td>
          <td>26.815704</td>
          <td>0.255156</td>
          <td>26.226339</td>
          <td>0.286794</td>
          <td>25.650276</td>
          <td>0.376592</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>31.659584</td>
          <td>4.225808</td>
          <td>27.084514</td>
          <td>0.227204</td>
          <td>26.617413</td>
          <td>0.135285</td>
          <td>25.852179</td>
          <td>0.112582</td>
          <td>26.012247</td>
          <td>0.240767</td>
          <td>25.911534</td>
          <td>0.459837</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>29.885107</td>
          <td>2.539141</td>
          <td>26.524773</td>
          <td>0.141478</td>
          <td>26.022961</td>
          <td>0.080461</td>
          <td>25.725396</td>
          <td>0.100774</td>
          <td>25.183078</td>
          <td>0.119019</td>
          <td>24.687743</td>
          <td>0.171414</td>
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
          <td>0.890625</td>
          <td>27.063840</td>
          <td>0.631241</td>
          <td>27.019043</td>
          <td>0.246201</td>
          <td>25.986978</td>
          <td>0.091670</td>
          <td>25.379522</td>
          <td>0.088030</td>
          <td>24.972243</td>
          <td>0.116253</td>
          <td>24.614069</td>
          <td>0.189166</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>29.981937</td>
          <td>2.749287</td>
          <td>28.861766</td>
          <td>0.945019</td>
          <td>27.475146</td>
          <td>0.322530</td>
          <td>27.620538</td>
          <td>0.552225</td>
          <td>26.355667</td>
          <td>0.367872</td>
          <td>26.072111</td>
          <td>0.594034</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.331040</td>
          <td>0.372376</td>
          <td>26.027213</td>
          <td>0.107988</td>
          <td>24.760933</td>
          <td>0.031702</td>
          <td>23.916287</td>
          <td>0.024774</td>
          <td>23.175094</td>
          <td>0.024421</td>
          <td>22.796445</td>
          <td>0.039470</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.223574</td>
          <td>0.651256</td>
          <td>27.279431</td>
          <td>0.293061</td>
          <td>26.954512</td>
          <td>0.354473</td>
          <td>26.298104</td>
          <td>0.373222</td>
          <td>25.107277</td>
          <td>0.303062</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>27.815935</td>
          <td>1.027373</td>
          <td>25.720379</td>
          <td>0.080900</td>
          <td>25.445803</td>
          <td>0.056842</td>
          <td>24.869316</td>
          <td>0.056085</td>
          <td>24.413576</td>
          <td>0.071199</td>
          <td>23.668011</td>
          <td>0.083464</td>
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
          <td>2.147172</td>
          <td>26.552787</td>
          <td>0.441033</td>
          <td>26.186732</td>
          <td>0.123904</td>
          <td>26.051155</td>
          <td>0.099037</td>
          <td>26.322195</td>
          <td>0.202736</td>
          <td>25.742886</td>
          <td>0.228756</td>
          <td>25.753117</td>
          <td>0.479481</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.896179</td>
          <td>0.223207</td>
          <td>27.177563</td>
          <td>0.254488</td>
          <td>26.311799</td>
          <td>0.197658</td>
          <td>26.263311</td>
          <td>0.343363</td>
          <td>25.415029</td>
          <td>0.364888</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>31.644601</td>
          <td>4.352756</td>
          <td>27.580837</td>
          <td>0.389800</td>
          <td>26.866752</td>
          <td>0.198190</td>
          <td>26.498590</td>
          <td>0.232959</td>
          <td>25.991616</td>
          <td>0.278417</td>
          <td>26.672260</td>
          <td>0.896184</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.947875</td>
          <td>1.127328</td>
          <td>27.232619</td>
          <td>0.300705</td>
          <td>26.632183</td>
          <td>0.165436</td>
          <td>25.694530</td>
          <td>0.119775</td>
          <td>25.630268</td>
          <td>0.210344</td>
          <td>25.434728</td>
          <td>0.379834</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.029184</td>
          <td>0.619880</td>
          <td>26.541786</td>
          <td>0.166488</td>
          <td>26.109061</td>
          <td>0.103054</td>
          <td>25.598422</td>
          <td>0.107766</td>
          <td>25.059771</td>
          <td>0.126684</td>
          <td>25.065989</td>
          <td>0.277772</td>
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
          <td>0.890625</td>
          <td>26.259859</td>
          <td>0.312048</td>
          <td>26.979864</td>
          <td>0.208257</td>
          <td>26.003642</td>
          <td>0.079111</td>
          <td>25.323721</td>
          <td>0.070750</td>
          <td>25.021415</td>
          <td>0.103378</td>
          <td>24.606505</td>
          <td>0.159965</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>33.057329</td>
          <td>5.608124</td>
          <td>28.496822</td>
          <td>0.670582</td>
          <td>30.123405</td>
          <td>1.595822</td>
          <td>27.913090</td>
          <td>0.594003</td>
          <td>26.765757</td>
          <td>0.438191</td>
          <td>27.696151</td>
          <td>1.429405</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.543435</td>
          <td>0.410432</td>
          <td>26.099595</td>
          <td>0.105054</td>
          <td>24.802578</td>
          <td>0.029676</td>
          <td>23.878444</td>
          <td>0.021597</td>
          <td>23.161696</td>
          <td>0.021828</td>
          <td>22.877858</td>
          <td>0.038137</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.500378</td>
          <td>1.406856</td>
          <td>27.658220</td>
          <td>0.393984</td>
          <td>27.049980</td>
          <td>0.380660</td>
          <td>25.873230</td>
          <td>0.265006</td>
          <td>25.303145</td>
          <td>0.352917</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.177434</td>
          <td>0.292320</td>
          <td>25.769596</td>
          <td>0.073236</td>
          <td>25.417983</td>
          <td>0.047135</td>
          <td>24.860576</td>
          <td>0.046975</td>
          <td>24.404118</td>
          <td>0.060047</td>
          <td>23.625726</td>
          <td>0.068038</td>
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
          <td>2.147172</td>
          <td>27.096410</td>
          <td>0.614939</td>
          <td>26.518609</td>
          <td>0.150555</td>
          <td>26.181251</td>
          <td>0.100066</td>
          <td>25.852821</td>
          <td>0.122222</td>
          <td>25.796148</td>
          <td>0.216759</td>
          <td>25.084870</td>
          <td>0.258123</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.375830</td>
          <td>0.345649</td>
          <td>26.831629</td>
          <td>0.186378</td>
          <td>26.874736</td>
          <td>0.171412</td>
          <td>26.342875</td>
          <td>0.174742</td>
          <td>26.582313</td>
          <td>0.385915</td>
          <td>25.450863</td>
          <td>0.326899</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.324025</td>
          <td>0.707457</td>
          <td>27.105662</td>
          <td>0.240701</td>
          <td>26.914300</td>
          <td>0.182902</td>
          <td>26.319849</td>
          <td>0.177073</td>
          <td>26.004277</td>
          <td>0.250395</td>
          <td>24.874961</td>
          <td>0.210750</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.451420</td>
          <td>0.798174</td>
          <td>27.728617</td>
          <td>0.416705</td>
          <td>26.556220</td>
          <td>0.143619</td>
          <td>25.985586</td>
          <td>0.142236</td>
          <td>25.859478</td>
          <td>0.236344</td>
          <td>25.814897</td>
          <td>0.473217</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.566896</td>
          <td>0.406636</td>
          <td>26.574006</td>
          <td>0.152565</td>
          <td>26.225435</td>
          <td>0.099965</td>
          <td>25.746199</td>
          <td>0.106882</td>
          <td>24.974609</td>
          <td>0.103136</td>
          <td>25.080260</td>
          <td>0.247446</td>
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
