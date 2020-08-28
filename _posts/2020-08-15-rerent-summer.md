---
title: Django and Summer Internship
mathjax: false
toc: true
categories:
  - development
tags:
  - update
---

For the past month and a half, I've been working  as a backend developer for [ReRent](https://www.rerent.co), a Yale SOM-based hospitality startup. Working alongside motivated, inspirational people of ReRent has been such a transformative experience me, which is why I decided to share some of that experience through this post. I decided that now is also good timing since we just pushed out an MVP a few days ago; now the team is on a brief hiatus before getting back to work for the fall. Reorientation phase, if you will. 

I have admittedly been slacking off these past few weeks with this blog, largely because I was spending a lot of my time learning Django, React, and many more. 

# Django

When I first came across Django a few months back through the PPETrackr project, I never understood how Django worked and why it was so powerful. As a volunteer who was exclusively focused on building out the dashboard with Plotly and Dash, I lacked the holistic understanding of how the Plotly dashboard blended into the bigger frame of the project that was Django. I was still more of a Flask fan than a Django fan---I was admittedly intimidated by its ostensible complexity.

Fast forward a few months and now here I am, a Django backend engineer intern who now loves Django to its fullest. Django is such a powerful battery-packed backend framework that offers so much functionality out of the box. Below are some of the features that I love about Django.

## ORM

Although I’ve seen people say otherwise, Django’s ORM is a powerful yet highly intuitive way of interacting with the database. It makes things like foreign keys, one-to-one, and many-to-many relationships so easy to deal with. In reality, forming a many-to-many relationship requires there to be an intermediate table that represents such a relationship. However, Django takes care of all that under the hood, lifting the burden off of the developer’s shoulders. 

In our web application, we had to implement a decently complex model with a chain of foreign keys and many-to-many relationships. In summary, a user could have multiple properties, and each property could have multiple trip plans associated with them. Simultaneously, we had staff users who would be assigned a number of zip codes. Each property, then, was associated with one of these zip codes. 
While this sounds like a decently complicated relationship, with Django querying becomes extremely easy. For example, to obtain all the locations of the user 



```python
request.user.rerenter_profile.location.all()
```



 This is assuming that the user is logged in, and that we have access to the `request` via some view function. In the case of class-based views, we would use `self.request` instead.

One feature I really like about Django ORM is the ability to query backwards, or upstream. Normally, we would get the `location` associated with some awayplan via a foreign key lookup. For instance, the following query would give us the location of the first awayplan in the database.

```python
AwayPlan.objects.get(id=1).location
```

This simply flows from the fact that the location is a foreign key attribute associated with the `AwayPlan` model. However, we can also query the other way around:

```python
user.rerenter_profile.location.filter(is_default=True).first().awayplan_set.all()
```

In this case, we get the first default location of the user, then obtain a `queryset` object which contains all the `awayplan` objects associated with that property. (Note that I’m using the word property interchangeably with location. We chose to use the word `location` in our code since `property` is a reserved keyword in Python.)

I thoroughly enjoyed writing these query statements because they felt more like little brain teaser puzzles. Of course, when the schema gets extremely complicated, so will the Django query, but nonetheless I still feel confident in the flexibility and the level of abstraction that the Django ORM offers.

## Views

Django’s views are surprisingly similar to those of Flask. The biggest difference is probably the fact that Django offers a lot of class-based views out of the box. Personally, I’m still not the hugest fan of class-based views, simply because I feel like it doesn’t provide quite enough the level of granularity I want, but it is a robust, powerful way of writing views no doubt. As I will explain in the next section on the REST framework, I do believe using class-based views makes a lot of sense in some contexts. 

### Decorators

Decorators are, in simple terms, ways of wrapping view functions to place restrictions on which users can and cannot access them. For example, we might only want to expose some endpoints to users who are logged in. In that case we might use a decorator like `@login_required`, which is unsurprisingly a function that is offered by Django right out of the box. 

In our case, we wanted to place more stringent restrictions on some views: not only should the user be logged in, but they should also be, for instance, a ReRenter or Staff to be able to access a certain URL. Otherwise, we want to show them an access denied page.



```python
def status_allowed(roles):
    def decorator(view):
        @login_required
        def wrapper(request, *args, **kwargs):
            if request.user.user_type in roles:
                return view(request, *args, **kwargs)
            raise Http400
        return wrapper
    return decorator
```



While this isn’t exactly the code that we used, it covers the gist of the idea: we use the `login_required` decorator, then apply the custom user type check decorator to make sure that only authorized users can access the view. 

## Migrations

Database migrations was something that really bite us hard in the beginning. The main reasoning was that we weren’t version controlling migration files at all; we had intentionally `.gitignore`d it, thinking that migration files are best kept out of our GitHub repository. 

Turns out that this was a very bad idea. Because the developers on the team each had their own migrations, changes to the database often messed up how Django dealt with migrations. It was even worse with deployment because the Heroku app was also having a lot of hiccups whenever we added a column to a table, for instance. In the process, we learned a bunch of Django commands like `fake` or `zero` migrations. 
After realizing that version controlling migration files took care of the issue, the development process sped up quite a bit. I also learned about `Makefile`, which gave access to a bunch of shortcut commands, almost like aliases, except that I didn’t have to clutter my `.bashrc` or `.bash_profile`. 



```
.PHONY=resetmigrations

resetmigrations:
    find . -path "*/migrations/*.py" -not -name "__init__.py" -delete
    find . -path "*/migrations/*.pyc" -delete
    rm db.sqlite3
```


With this in the directory, we can now run `make resetmigrations` to get rid of all migration files as well as the database. This is a destructive operation and should be avoided, but when drastic schema changes are being made in a development environment, I found it to be quite useful. All credits go to [Mitesh](https://github.com/oxalorg).

# REST

After the MVP release, I began looking more into frontend frameworks, most notably React.js. And as I was taking a deeper dive into how React works (and React is a cool framework by the way, I might write about it in the near future), I got to know more about how the backend and frontend interact with each other. 

## Templating

Django does not require one to have a lot of knowledge of frontend or JavaScript in general, at least the way it is set up right out of the box. Instead, it comes with DTL, or the Django Templating Language, which is a convenient way of rendering context variables passed over from the backend. For example, in a view function, we might say something like



```python
render("template_name.html", {"some_var": some_var})
```



The first argument is the template to be rendered, and the second argument in the form of a dictionary is the context. The variables in the context can be accessed in the templates via `{{ some_var }}`.

Instead of DTL, I prefer to use Jinja2, just because the Jinja2 syntax is a bit more Pythonic, and also because Jinja2’s rendering speed is superior to that of the default Django Templating backend. To read more about specific benchmarks, I recommend that you check out [this post](https://www.dyspatch.io/blog/python-templating-performance-showdown-django-vs-jinja/). The takeaway is that Jinja2 is probably a better choice, although it does require some more configuration to work as smoothly as Django. It might also be somewhat frustrating because there are lesser tutorials on how to get Jinja2 working with third party packages like django-crispy-forms.

## RESTful API

I’ve seen the word REST being thrown around for a while, but it is these past few weeks that I really started to wrap my head around the concept. As I’m still in the learning stage of things, the explanations that follow may have inaccuracies and deficiencies, yet it is my hope that it would provide a beginner-friendly introduction to the idea of a REST API.

A REST API can loosely be understood as a set of endpoints that allow users to make HTTP requests, such as but not limited to get and post. In other words, the REST API is an entrypoint through which users can interact with the database. Of course, a robust API would not allow users to do whatever they want with the database; instead, the API---which is effectively a web application---would make sure that the user is permitted to make specific requests. 

The REST API responds to user requests by sending a JSON response. JSON stands short for "JavaScript Serializable Object Notation." If you’ve worked with JavaScript before, you’ll probably be familiar with JavaScript objects, which are basically what JSON is. If you’re more of a Pythonista like I am, just think of dictionaries. JSON is a way of serializing data, which works well with browsers and HTTP requests. 

The REST API provides a way for the backend to communicate with the frontend. The user wouldn’t be directly accessing the endpoints of the REST API, since, as stated earlier, the REST API’s way of responding to requests is by returning some JSON. Therefore, the job of the frontend is to render that JSON response in a user-friendly visual interface. And that’s there all is to it: when a user presses a button, the frontend "pings" the backend, and the backend responds with a JSON object. The frontend then displays the JSON object, whether it be a list of tweets, a user authentication token, or some error message. 

By default, Django does not provide a RESTful API in the sense that there is no complete decoupling between the frontend and the backend. The backend communicates directly with the frontend via DTL. The frontend gains limited interactivity with the backend via things like form submissions. 

The good news is that Django has so much community development support that whatever you’re looking for, there’s always going to be a Django package for it. In the case of RESTifying Django, there is an incredibly widely used, industry-standard package known as the Django Rest Framework, which is often abbreviated as DRF. DRF allows Django developers to easily build up a RESTful API with only a few minor tweaks. 

And this is where my earlier point on function and class-based views comes in. In the case of vanilla Django, the view function might have to deal with some aspects of the frontend since we need to be careful with context variables. In the case of a REST API, however, the heavy lifting is all down by the frontend, which is why the backend can be streamlined quite a bit. In those instances, the default class-based views can be very convenient. For example, here is an example of a class view taken from the [DRF docs](https://www.django-rest-framework.org).



```python
class ItemList(APIView):
    def get(self, request, format=None):
        items = Item.objects.all()
        serializer = ItemSerializer(items, many=True)
    	return Response(serializer.data)

    def post(self, request, format=None):
        serializer = ItemSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
        	return Response(serializer.data, status=status.HTTP_201_CREATED)
    	return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
```



At a glance, this doesn’t seem different from a function-based view at all. Well, the reason is that we didn’t make use of higher level abstractions that DRF provides, just like Django itself. If we use some `mixins`, for instance, we can simplify the code above to something as follows:



```python
class ItemList(mixins.ListModelMixin,
               mixins.CreateModelMixin,
               generics.GenericAPIView):
    queryset = Item.objects.all()
    serializer_class = ItemSerializer

    def get(self, request, *args, **kwargs):
        return self.list(request, *args, **kwargs)

    def post(self, request, *args, **kwargs):
        return self.create(request, *args, **kwargs)
```



The `mixins.ListModelMixin`, as the name implies, provides a `self.list()` method, which is essentially what we want to show upon a HTTP get request by the client. Similarly, the `self.create()` method is made available by inheriting from the `mixins.CreateModelMixin`. As you can see, these functionalities are something that only inheritance and object-oriented programming with classes can provide; function-based views cannot and do not provide this level of abstraction and convenience. 



```python
class ItemList(generics.ListCreateAPIView):
	queryset = Item.objects.all()
	serializer_class = ItemSerializer
```



And that is all we need! Just like Django, DRF is filled with features that make building a REST API but a simple task.

I’d like to add one last bit of DRF’s magic: viewsets and routers. Viewsets provide an additional layer of abstraction even farther on top of what DRF class-based views provide. An easy way to understand viewsets is to view them as a collection of different views. Then, the router automatically maps these views with each url pattern. 



```python
class ItemViewSet(viewsets.ModelViewSet):
    serializer_class = ItemSerializer
    queryset = Item.objects.all()
```



Then, we can set up the router as follows:

```python
router = DefaultRouter()
router.register(r'items', ItemViewSet, basename='item')
urlpatterns = router.urls

```



I’m still not entirely sure how this automatica mapping works---that will require me to look at the DRF source code---but it is helpful to know that there are even higher levels of abstraction that DRF provides. I personally don’t think I’ll be using viewsets and routers that much because its level of abstraction takes away a lot of room for customization and micro-optimization, yet it is a useful feature no doubt, especially if all you need is to build a quick REST API.

# Conclusion

I was originally planning to write about things like Agile and React, but realized that this post was getting a bit too long. While combining DRF with React will be an interesting engineering challenge, we will probably be spending more time debugging and improving the MVP, which is a vanilla Django app with a Jinja2 templating backend as it stands. 

In general, working in a collaborative environment on a product has been a delightful challenge. Although it is disappointing that I haven’t been able to write more on this blog on what have been my mainstream topics so far---math and machine learning---I hope to continue writing in some fashion, as I consider writing to be an incredibly rewarding, creative endeavor that also allows me to reflect, improve, and contemplate. 

Thanks for reading, and catch you up in the next one!